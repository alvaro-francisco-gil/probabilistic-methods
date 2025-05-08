from typing import List, Set, Dict, Tuple, Optional, Any
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
        steps = ["Initial factors created"]
        factors = self._initial_factors()
        # reduce evidence
        for var, val in evidence.items():
            factors = [f.reduce(var, val) if var in f.scope else f for f in factors]
            steps.append(f"Reduced factors on {var}={val}")
        # merge smallest-scope pairs
        while len(factors) > 1:
            best = None
            for f1, f2 in itertools.combinations(factors, 2):
                cost = len(set(f1.scope + f2.scope))
                if best is None or cost < best[0]:
                    best = (cost, f1, f2)
            _, f1, f2 = best
            steps.append(f"Grouping factors {f1.scope} and {f2.scope}")
            merged = f1.multiply(f2)
            factors = [f for f in factors if f not in (f1, f2)] + [merged]
            steps.append(f"Merged to factor {merged.scope}")
        # finalize
        result = factors[0].normalize()
        steps.append("Normalize final grouped factor")
        idx = result.scope.index(query)
        prob = sum(p for a, p in result.table.items() if a[idx])
        return prob, steps

    def join_tree(self) -> List[str]:
        steps = [
            "Moralize network: connect all co-parents",
            "Triangulate: add fill-in edges by min-fill heuristic",
            "Identify maximal cliques in chordal graph",
            "Build clique tree via maximum spanning tree on sepset sizes",
            "Assign CPTs to cliques covering their scope",
            "Initialize potentials and pass messages for propagation"
        ]
        return steps

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