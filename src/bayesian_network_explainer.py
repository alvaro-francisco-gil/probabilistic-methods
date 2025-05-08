from typing import List, Set, Dict, Tuple, Optional, Any
import networkx as nx
from src.bayesian_network import BayesianNetwork
import itertools

class Factor:
    """
    A factor over a set of variables with a probability table.
    Scope: ordered list of variable names.
    Table: dict mapping assignments (tuples of bools) to probabilities.
    """
    def __init__(self, scope: List[str], table: Dict[Tuple[bool, ...], float]):
        self.scope = list(scope)
        self.table = dict(table)

    def reduce(self, var: str, value: bool) -> 'Factor':
        if var not in self.scope:
            return self
        idx = self.scope.index(var)
        new_scope = [v for v in self.scope if v != var]
        new_table = {}
        for assignment, prob in self.table.items():
            if assignment[idx] == value:
                new_assignment = tuple(v for i, v in enumerate(assignment) if i != idx)
                new_table[new_assignment] = prob
        return Factor(new_scope, new_table)

    def marginalize(self, var: str) -> 'Factor':
        if var not in self.scope:
            return self
        idx = self.scope.index(var)
        new_scope = [v for v in self.scope if v != var]
        new_table: Dict[Tuple[bool, ...], float] = {}
        for assignment, prob in self.table.items():
            new_assignment = tuple(v for i, v in enumerate(assignment) if i != idx)
            new_table[new_assignment] = new_table.get(new_assignment, 0.0) + prob
        return Factor(new_scope, new_table)

    def multiply(self, other: 'Factor') -> 'Factor':
        new_scope = []
        for v in self.scope + other.scope:
            if v not in new_scope:
                new_scope.append(v)
        idx_self = [new_scope.index(v) for v in self.scope]
        idx_other = [new_scope.index(v) for v in other.scope]
        new_table: Dict[Tuple[bool, ...], float] = {}
        for assignment in itertools.product([False, True], repeat=len(new_scope)):
            key_self = tuple(assignment[i] for i in idx_self)
            key_other = tuple(assignment[i] for i in idx_other)
            new_table[assignment] = self.table.get(key_self, 0.0) * other.table.get(key_other, 0.0)
        return Factor(new_scope, new_table)

    def normalize(self) -> 'Factor':
        total = sum(self.table.values())
        if total == 0:
            return self
        new_table = {a: p / total for a, p in self.table.items()}
        return Factor(self.scope, new_table)

    def __repr__(self) -> str:
        return f"Factor(scope={self.scope}, table={self.table})"

class InferenceEngine:
    def __init__(self, bn: BayesianNetwork): self.bn = bn
    def _initial_factors(self) -> List[Factor]:
        factors = []
        for var in self.bn.variables:
            scope = [var] + self.bn.get_parents(var)
            table = {tuple(entry[v] for v in scope): entry['prob']
                     for entry in self.bn.get_cpt(var)}
            factors.append(Factor(scope, table))
        return factors

    def variable_elimination(self, query: str, evidence: Dict[str, bool]) -> Tuple[float, List[str]]:
        steps = []
        factors = self._initial_factors()
        # 1. reduce evidence
        for var, val in evidence.items():
            new_factors = []
            for f in factors:
                if var in f.scope:
                    steps.append(f"Reduce factor {f.scope} on {var}={val}")
                    new_factors.append(f.reduce(var, val))
                else:
                    new_factors.append(f)
            factors = new_factors
        # 2. elimination order: min-degree heuristic
        hidden = [v for v in self.bn.variables if v not in evidence and v != query]
        order = sorted(hidden, key=lambda v: sum(v in f.scope for f in factors))
        steps.append(f"Elimination order: {order}")
        # 3. eliminate
        for var in order:
            group = [f for f in factors if var in f.scope]
            for f in group:
                steps.append(f"Selected factor {f.scope} for eliminating {var}")
            prod = group[0]
            for f in group[1:]:
                prod = prod.multiply(f)
            steps.append(f"Multiply to get factor {prod.scope}")
            marg = prod.marginalize(var)
            steps.append(f"Marginalize out {var} to get factor {marg.scope}")
            factors = [f for f in factors if f not in group] + [marg]
        # 4. final multiply & normalize
        result = factors[0]
        for f in factors[1:]:
            result = result.multiply(f)
        steps.append(f"Multiply remaining to get factor {result.scope}")
        norm = result.normalize()
        steps.append("Normalize final factor")
        idx = norm.scope.index(query)
        prob = sum(p for a, p in norm.table.items() if a[idx])
        return prob, steps

    def arc_inversion(self, query: str, evidence: Dict[str, bool]) -> Tuple[float, List[str]]:
        steps = []
        factors = self._initial_factors()
        for var, val in evidence.items():
            new_factors = []
            for f in factors:
                if var in f.scope:
                    steps.append(f"Reduce factor {f.scope} on {var}={val}")
                    new_factors.append(f.reduce(var, val))
                else:
                    new_factors.append(f)
            factors = new_factors

        path = self._find_path(query, list(evidence.keys())[0])
        steps.append(f"Path for inversion: {path}")

        for i in range(len(path)-1):
            parent, child = path[i], path[i+1]
            steps.append(f"Invert arc {parent}->{child}")
            related = [f for f in factors if parent in f.scope or child in f.scope]
            if not related:
                raise ValueError(f"No factors contain {parent} or {child} for arc inversion.")
            combined = related[0]
            for f in related[1:]:
                combined = combined.multiply(f)
            steps.append(f"Combined factors for arc {parent}->{child}: scope={combined.scope}")
            factors = [f for f in factors if f not in related] + [combined]

        joint = factors[0]
        for f in factors[1:]:
            joint = joint.multiply(f)
        steps.append(f"Multiply inverted factors to get factor {joint.scope}")
        norm = joint.normalize()
        steps.append("Normalize to get posterior")
        idx = norm.scope.index(query)
        prob = sum(p for a, p in norm.table.items() if a[idx])
        return prob, steps

    def grouping(self, query: str, evidence: Dict[str, bool]) -> Tuple[float, List[str]]:
        steps = ["Initializing factors from CPTs"]
        factors = self._initial_factors()
        
        # Incorporate evidence
        for var, val in evidence.items():
            factors = [f.reduce(var, val) if var in f.scope else f for f in factors]
            steps.append(f"Reduced factors on evidence {var}={val}")
        
        # Step 1: Moralize the graph
        steps.append("Creating moral graph by connecting co-parents")
        moral_graph = nx.Graph()
        for node in self.bn.G.nodes():
            moral_graph.add_node(node)
        for u, v in self.bn.G.edges():
            moral_graph.add_edge(u, v)
        for node in self.bn.G.nodes():
            parents = list(self.bn.G.predecessors(node))
            for i in range(len(parents)):
                for j in range(i+1, len(parents)):
                    moral_graph.add_edge(parents[i], parents[j])
        
        # Step 2: Triangulate the graph
        steps.append("Triangulating graph using min-fill heuristic")
        triangulated = moral_graph.copy()
        remaining = list(triangulated.nodes())
        while remaining:
            # Find node with minimum fill
            min_fill = float('inf')
            best_node = None
            for node in remaining:
                nbrs = list(triangulated.neighbors(node))
                fill = sum(1 for i, j in itertools.combinations(nbrs, 2)
                        if not triangulated.has_edge(i, j))
                if fill < min_fill:
                    min_fill = fill
                    best_node = node
            
            # Add fill edges
            nbrs = list(triangulated.neighbors(best_node))
            for i, j in itertools.combinations(nbrs, 2):
                if not triangulated.has_edge(i, j):
                    triangulated.add_edge(i, j)
            
            remaining.remove(best_node)
        
        # Step 3: Find maximal cliques
        steps.append("Finding maximal cliques in triangulated graph")
        cliques = list(nx.find_cliques(triangulated))
        steps.append(f"Found {len(cliques)} cliques")
        
        # Step 4: Build junction tree
        steps.append("Building junction tree via maximum spanning tree")
        jt = nx.Graph()
        for i, clique in enumerate(cliques):
            jt.add_node(i, vars=set(clique))
        
        # Add edges with weights based on intersection size
        for i, j in itertools.combinations(range(len(cliques)), 2):
            intersection = set(cliques[i]) & set(cliques[j])
            if intersection:
                jt.add_edge(i, j, weight=len(intersection), sepset=intersection)
        
        # Create maximum spanning tree
        mst = nx.maximum_spanning_tree(jt, weight='weight')
        
        # Step 5: Assign factors to cliques
        steps.append("Assigning factors to cliques")
        potentials = {i: Factor([], {(): 1.0}) for i in range(len(cliques))}
        for factor in factors:
            scope_set = set(factor.scope)
            # Find smallest containing clique
            best_clique = None
            best_size = float('inf')
            for i, clique in enumerate(cliques):
                if scope_set.issubset(set(clique)) and len(clique) < best_size:
                    best_clique = i
                    best_size = len(clique)
            
            if best_clique is not None:
                potentials[best_clique] = potentials[best_clique].multiply(factor)
                steps.append(f"Assigned factor {factor.scope} to clique {best_clique}")
        
        # Step 6: Perform message passing
        steps.append("Performing belief propagation on junction tree")
        root = 0  # Choose arbitrary root
        
        def collect_messages(node, parent=None):
            for neighbor in mst.neighbors(node):
                if neighbor != parent:
                    collect_messages(neighbor, node)
                    
                    # Calculate message from neighbor to node
                    sepset = mst[node][neighbor]['sepset']
                    message = potentials[neighbor]
                    
                    # Marginalize out variables not in separator
                    for var in list(message.scope):
                        if var not in sepset:
                            message = message.marginalize(var)
                    
                    # Update parent's potential
                    potentials[node] = potentials[node].multiply(message)
                    steps.append(f"Passed message from clique {neighbor} to {node}")
        
        # Run collection phase
        collect_messages(root)
        
        # Step 7: Compute query result
        steps.append(f"Computing marginal for {query}")
        query_clique = None
        for i, clique in enumerate(cliques):
            if query in clique:
                query_clique = i
                break
        
        if query_clique is None:
            raise ValueError(f"Query variable {query} not found in any clique")
        
        # Marginalize to get query distribution
        marginal = potentials[query_clique]
        for var in list(marginal.scope):
            if var != query:
                marginal = marginal.marginalize(var)
        
        # Normalize
        marginal = marginal.normalize()
        
        # Get probability for query=True
        prob_true = sum(prob for assignment, prob in marginal.table.items() 
                    if assignment[0])  # Assuming query is the first variable
        
        return prob_true, steps

    def _find_path(self, start: str, end: str) -> List[str]:
        visited = set()
        path: List[str] = []
        def dfs(u: str) -> bool:
            if u == end:
                path.append(u)
                return True
            visited.add(u)
            for v in self.bn.get_children(u) + self.bn.get_parents(u):
                if v not in visited and dfs(v):
                    path.append(u)
                    return True
            return False
        dfs(start)
        return list(reversed(path))