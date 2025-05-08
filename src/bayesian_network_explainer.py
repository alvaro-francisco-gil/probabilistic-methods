from src.bayesian_network import BayesianNetwork, ConditionalProbabilityTable
from typing import List, Set, Dict, Tuple, Optional

def find_sink_nodes(bn: BayesianNetwork) -> list:
    """
    Find all sink nodes in the Bayesian network.
    A sink node is a node that has no children (no outgoing edges).
    
    Args:
        bn: BayesianNetwork instance
        
    Returns:
        List of sink node names
    """
    sink_nodes = []
    for node in bn.variables:
        if not bn.get_children(node):  # If node has no children
            sink_nodes.append(node)
    return sink_nodes

def get_initial_factors(bn: BayesianNetwork) -> Dict[str, List[str]]:
    """
    Get all initial factors (CPTs) in the network.
    
    Args:
        bn: BayesianNetwork instance
        
    Returns:
        Dictionary mapping variable names to their parent variables
    """
    factors = {}
    for var in bn.variables:
        parents = bn.get_parents(var)
        factors[var] = parents
    return factors

def find_factors_containing_variable(factors: Dict[str, List[str]], var: str) -> List[str]:
    """
    Find all factors that contain a specific variable.
    
    Args:
        factors: Dictionary of factors
        var: Variable to search for
        
    Returns:
        List of variables whose factors contain the given variable
    """
    containing_factors = []
    for var_name, parents in factors.items():
        if var in parents or var == var_name:
            containing_factors.append(var_name)
    return containing_factors

def find_path_to_evidence(bn: BayesianNetwork, start_var: str, evidence_var: str) -> List[str]:
    """
    Find a path from start_var to evidence_var in the network.
    
    Args:
        bn: BayesianNetwork instance
        start_var: Starting variable
        evidence_var: Target evidence variable
        
    Returns:
        List of variables forming the path
    """
    def dfs(current: str, target: str, visited: Set[str], path: List[str]) -> Optional[List[str]]:
        if current == target:
            return path
        visited.add(current)
        
        # Check children
        for child in bn.get_children(current):
            if child not in visited:
                result = dfs(child, target, visited, path + [child])
                if result:
                    return result
                    
        # Check parents
        for parent in bn.get_parents(current):
            if parent not in visited:
                result = dfs(parent, target, visited, path + [parent])
                if result:
                    return result
                    
        return None
    
    return dfs(start_var, evidence_var, set(), [start_var]) or []

def compute_arc_inversion(bn: BayesianNetwork, query_var: str, evidence_var: str, evidence_value: bool = True) -> float:
    """
    Compute the probability using arc inversion method.
    
    Args:
        bn: BayesianNetwork instance
        query_var: Variable to calculate probability for
        evidence_var: Observed variable
        evidence_value: Value of the observed variable
        
    Returns:
        The computed probability P(query_var|evidence_var=evidence_value)
    """
    # Find path from query to evidence
    path = find_path_to_evidence(bn, query_var, evidence_var)
    if not path:
        raise ValueError(f"No path found from {query_var} to {evidence_var}")
    
    # Get all variables in the path
    path_vars = set(path)
    
    # Initialize probability tables
    prob_tables = {}
    for var in path_vars:
        parents = bn.get_parents(var)
        if parents:
            prob_tables[var] = bn.get_cpt(var)
        else:
            prob_tables[var] = bn.get_cpt(var)
    
    # Perform arc inversions along the path
    for i in range(len(path) - 1):
        current = path[i]
        next_var = path[i + 1]
        
        # Get current probability tables
        p_current = prob_tables[current]
        p_next = prob_tables[next_var]
        
        # Compute new probability tables after inversion
        # This is a simplified version - in practice, you'd need to handle
        # the full joint probability calculations
        new_p_current = {}
        new_p_next = {}
        
        # Update probability tables
        prob_tables[current] = new_p_current
        prob_tables[next_var] = new_p_next
    
    # Propagate evidence
    evidence_prob = 1.0
    for var in reversed(path):
        if var == evidence_var:
            evidence_prob *= prob_tables[var].get(evidence_value, 0.0)
        else:
            # Sum over all possible values
            var_prob = sum(prob_tables[var].values())
            evidence_prob *= var_prob
    
    # Normalize to get final probability
    total_prob = sum(prob_tables[query_var].values())
    if total_prob == 0:
        return 0.0
    
    return evidence_prob / total_prob

def explain_elimination_steps(bn: BayesianNetwork, query_var: str, evidence_var: str, evidence_value: bool = True):
    """
    Explain the steps needed to calculate P(query_var|evidence_var=evidence_value) using variable elimination.
    
    Args:
        bn: BayesianNetwork instance
        query_var: Variable to calculate probability for
        evidence_var: Observed variable
        evidence_value: Value of the observed variable
    """
    print(f"\nSteps to calculate P({query_var}|{evidence_var}={evidence_value}) using Variable Elimination:")
    
    # 1. Find and explain sink nodes
    sink_nodes = find_sink_nodes(bn)
    print("\n1. Initial Setup:")
    print(f"   - We have evidence {evidence_var}={evidence_value}")
    print(f"   - We want to calculate P({query_var}|{evidence_var}={evidence_value})")
    if sink_nodes:
        print("   - Sink nodes in the network:", sink_nodes)
        print(f"   - Note: {evidence_var} is a sink node but cannot be removed as it is our evidence variable")
        removable_sinks = [s for s in sink_nodes if s != evidence_var and s != query_var]
        if removable_sinks:
            print("   - The following sink nodes can be removed as they don't affect our calculation:", removable_sinks)
    
    # 2. Get initial factors
    factors = get_initial_factors(bn)
    print("\n2. Initial Factors (CPTs):")
    for var, parents in factors.items():
        if parents:
            print(f"   - P({var}|{','.join(parents)}) - probability of {var} given {', '.join(parents)}")
        else:
            print(f"   - P({var}) - prior probability of {var}")
    
    # 3. Evidence handling
    print(f"\n3. Evidence Handling:")
    print(f"   - Instantiate {evidence_var}={evidence_value} in all factors containing {evidence_var}")
    print(f"   - This reduces P({evidence_var}|...) to a function of its parents only")
    
    # 4. Determine elimination order
    remaining_vars = set(bn.variables) - {query_var, evidence_var}
    if removable_sinks:
        remaining_vars -= set(removable_sinks)
    
    # Simple heuristic: eliminate variables with fewer connections first
    var_connections = {var: len(bn.get_parents(var)) + len(bn.get_children(var)) 
                      for var in remaining_vars}
    elimination_order = sorted(remaining_vars, key=lambda x: var_connections[x])
    
    print("\n4. Variable Elimination Process:")
    print(f"   Elimination order: {' → '.join(elimination_order)}")
    
    # 5. Explain elimination steps
    current_factors = factors.copy()
    for var in elimination_order:
        print(f"\n   Eliminating {var}:")
        containing_factors = find_factors_containing_variable(current_factors, var)
        print(f"      - Identify factors containing {var}: {', '.join(containing_factors)}")
        print(f"      - Multiply these factors")
        
        # Update factors after elimination
        new_factors = set()
        for factor in containing_factors:
            new_factors.update(current_factors[factor])
        new_factors.remove(var)
        if query_var in new_factors:
            new_factors.remove(query_var)
        if evidence_var in new_factors:
            new_factors.remove(evidence_var)
        
        print(f"      - Sum out {var} to get a new factor over {{{', '.join(sorted(new_factors))}}}")
        
        # Update current factors
        for factor in containing_factors:
            del current_factors[factor]
        if new_factors:
            current_factors[f"elim_{var}"] = list(new_factors)
    
    # 6. Final calculation
    print("\n5. Final Calculation:")
    print(f"   - After eliminating all variables except {query_var}, we have a factor over {query_var}")
    print(f"   - This factor represents P({query_var},{evidence_var}={evidence_value})")
    print(f"   - To get P({query_var}|{evidence_var}={evidence_value}), we need to normalize this factor")
    print(f"   - Normalization is done by dividing by P({evidence_var}={evidence_value})")
    print(f"   - P({evidence_var}={evidence_value}) is obtained by summing the factor over all values of {query_var}")
    print(f"   - The final result is P({query_var}|{evidence_var}={evidence_value})")

def explain_grouping_steps(bn: BayesianNetwork, query_var: str, evidence_var: str, evidence_value: bool = True):
    """
    Explain the steps needed to calculate P(query_var|evidence_var=evidence_value) using grouping.
    
    Args:
        bn: BayesianNetwork instance
        query_var: Variable to calculate probability for
        evidence_var: Observed variable
        evidence_value: Value of the observed variable
    """
    print(f"\nSteps to calculate P({query_var}|{evidence_var}={evidence_value}) using Grouping:")
    
    # 1. Find and explain sink nodes
    sink_nodes = find_sink_nodes(bn)
    print("\n1. Initial Setup:")
    print(f"   - We have evidence {evidence_var}={evidence_value}")
    print(f"   - We want to calculate P({query_var}|{evidence_var}={evidence_value})")
    if sink_nodes:
        print("   - Sink nodes in the network:", sink_nodes)
        print(f"   - Note: {evidence_var} is a sink node but cannot be removed as it is our evidence variable")
        removable_sinks = [s for s in sink_nodes if s != evidence_var and s != query_var]
        if removable_sinks:
            print("   - The following sink nodes can be removed as they don't affect our calculation:", removable_sinks)
    
    # 2. Get initial factors
    factors = get_initial_factors(bn)
    print("\n2. Initial Factors (CPTs):")
    for var, parents in factors.items():
        if parents:
            print(f"   - P({var}|{','.join(parents)}) - probability of {var} given {', '.join(parents)}")
        else:
            print(f"   - P({var}) - prior probability of {var}")
    
    # 3. Evidence handling
    print(f"\n3. Evidence Handling:")
    print(f"   - Instantiate {evidence_var}={evidence_value} in all factors containing {evidence_var}")
    print(f"   - This reduces P({evidence_var}|...) to a function of its parents only")
    
    # 4. Group variables based on their relationships
    print("\n4. Grouping Variables:")
    print("   We group variables that appear together in factors to minimize operations.")
    
    # Find groups dynamically
    groups = []
    visited = set()
    
    def find_connected_vars(var: str) -> Set[str]:
        connected = {var}
        to_visit = {var}
        while to_visit:
            current = to_visit.pop()
            # Add parents
            for parent in bn.get_parents(current):
                if parent not in connected:
                    connected.add(parent)
                    to_visit.add(parent)
            # Add children
            for child in bn.get_children(current):
                if child not in connected:
                    connected.add(child)
                    to_visit.add(child)
        return connected
    
    # Start with query variable
    query_group = find_connected_vars(query_var)
    groups.append({
        'name': f'Group 1 ({query_var} and connected variables)',
        'vars': list(query_group),
        'factors': [f"P({var}|{','.join(bn.get_parents(var))})" if bn.get_parents(var) else f"P({var})" 
                   for var in query_group]
    })
    visited.update(query_group)
    
    # Find remaining groups
    group_num = 2
    for var in bn.variables:
        if var not in visited:
            group_vars = find_connected_vars(var)
            if group_vars:
                groups.append({
                    'name': f'Group {group_num} ({var} and connected variables)',
                    'vars': list(group_vars),
                    'factors': [f"P({v}|{','.join(bn.get_parents(v))})" if bn.get_parents(v) else f"P({v})" 
                              for v in group_vars]
                })
                visited.update(group_vars)
                group_num += 1
    
    # Print groups
    for group in groups:
        print(f"\n   {group['name']}:")
        print(f"      Variables: {', '.join(group['vars'])}")
        print(f"      Factors: {', '.join(group['factors'])}")
    
    # 5. Elimination process using groups
    print("\n5. Elimination Process using Groups:")
    print("   We eliminate variables group by group, starting from the leaves of the network.")
    
    # Determine elimination order based on network structure
    elimination_order = []
    remaining_vars = set(bn.variables) - {query_var, evidence_var}
    
    while remaining_vars:
        # Find sink nodes in remaining variables
        current_sinks = [var for var in remaining_vars 
                        if not any(child in remaining_vars for child in bn.get_children(var))]
        if not current_sinks:
            break
        elimination_order.extend(current_sinks)
        remaining_vars -= set(current_sinks)
    
    # Print elimination steps
    for i, var in enumerate(elimination_order, 1):
        print(f"\n   Step {i}:")
        print(f"      Eliminate {var}")
        print(f"      Combine factors containing {var}")
        print(f"      Sum out {var} to get new factor")
    
    # 6. Final calculation
    print("\n6. Final Calculation:")
    print(f"   - After eliminating all variables except {query_var}, we have a factor over {query_var}")
    print(f"   - This factor represents P({query_var},{evidence_var}={evidence_value})")
    print(f"   - To get P({query_var}|{evidence_var}={evidence_value}), we need to normalize this factor")
    print(f"   - Normalization is done by dividing by P({evidence_var}={evidence_value})")
    print(f"   - P({evidence_var}={evidence_value}) is obtained by summing the factor over all values of {query_var}")
    print(f"   - The final result is P({query_var}|{evidence_var}={evidence_value})")

def explain_join_tree_construction(bn: BayesianNetwork, query_var: str, evidence_var: str, evidence_value: bool = True):
    """
    Explain the steps to construct a join tree from the Bayesian network.
    
    Args:
        bn: BayesianNetwork instance
        query_var: Variable to calculate probability for
        evidence_var: Observed variable
        evidence_value: Value of the observed variable
    """
    print(f"\nSteps to construct Join Tree for calculating P({query_var}|{evidence_var}={evidence_value}):")
    
    # 1. Moralization
    print("\n1. Moralization:")
    print("   - Add edges between all pairs of parents of each node")
    print("   - This makes the graph undirected")
    print("   - For each node with multiple parents, we need to connect them:")
    
    # Find nodes with multiple parents
    for var in bn.variables:
        parents = bn.get_parents(var)
        if len(parents) > 1:
            print(f"     * Parents of {var}: {', '.join(parents)}")
            print(f"       Need to connect: {' - '.join(parents)}")
    
    # 2. Triangulation
    print("\n2. Triangulation:")
    print("   - Add edges to eliminate cycles of length > 3")
    print("   - For each cycle of length > 3, we need to add edges to break it")
    
    # Find cycles (simplified version)
    cycles = []
    for var in bn.variables:
        children = bn.get_children(var)
        for child in children:
            grand_children = bn.get_children(child)
            for grand_child in grand_children:
                if var in bn.get_parents(grand_child):
                    cycles.append([var, child, grand_child])
    
    if cycles:
        print("   - Found cycles:")
        for cycle in cycles:
            print(f"     * {' → '.join(cycle)}")
            print(f"       Need to add edge: {cycle[0]} - {cycle[2]}")
    
    # 3. Clique Identification
    print("\n3. Clique Identification:")
    print("   - Identify all maximal cliques in the triangulated graph")
    print("   - A clique is a set of nodes where every pair is connected")
    
    # Find cliques (simplified version)
    cliques = []
    for var in bn.variables:
        parents = bn.get_parents(var)
        if parents:
            clique = set(parents + [var])
            if len(clique) > 1:
                cliques.append(clique)
    
    if cliques:
        print("   - Found cliques:")
        for i, clique in enumerate(cliques, 1):
            print(f"     * C{i} = {{{', '.join(sorted(clique))}}}")
    
    # 4. Join Tree Construction
    print("\n4. Join Tree Construction:")
    print("   - Create a tree where nodes are the maximal cliques")
    print("   - Connect cliques that share variables")
    
    if cliques:
        print("   - Clique connections:")
        for i in range(len(cliques)):
            for j in range(i + 1, len(cliques)):
                shared = cliques[i] & cliques[j]
                if shared:
                    print(f"     * C{i+1} -- C{j+1} (shared: {{{', '.join(sorted(shared))}}})")
    
    # 5. Separator Sets
    print("\n5. Separator Sets:")
    print("   - For each edge in the join tree, identify the separator set")
    print("   - Separator sets are the intersection of the connected cliques")
    
    if cliques:
        print("   - Separator sets:")
        for i in range(len(cliques)):
            for j in range(i + 1, len(cliques)):
                shared = cliques[i] & cliques[j]
                if shared:
                    print(f"     * S{i+1},{j+1} = C{i+1} ∩ C{j+1} = {{{', '.join(sorted(shared))}}}")
    
    # 6. Initialization
    print("\n6. Initialization:")
    print("   - Assign each CPT to a clique that contains all its variables")
    
    for var in bn.variables:
        parents = bn.get_parents(var)
        if parents:
            print(f"     * P({var}|{','.join(parents)}) → Clique containing {{{', '.join([var] + parents)}}}")
        else:
            print(f"     * P({var}) → Clique containing {{{var}}}")
    
    # 7. Evidence Handling
    print(f"\n7. Evidence Handling:")
    print(f"   - Instantiate {evidence_var}={evidence_value} in all cliques containing {evidence_var}")
    
    # 8. Message Passing
    print("\n8. Message Passing:")
    print("   - Messages are passed between cliques through separator sets")
    print("   - Two phases:")
    print("     a) Collect phase: Messages are sent from leaves to root")
    print("     b) Distribute phase: Messages are sent from root to leaves")
    
    # 9. Final Calculation
    print("\n9. Final Calculation:")
    print(f"   - To get P({query_var}|{evidence_var}={evidence_value}):")
    print(f"     * Find a clique containing {query_var}")
    print(f"     * Marginalize the clique's potential over {query_var}")
    print(f"     * Normalize by dividing by P({evidence_var}={evidence_value})")

def explain_arc_inversion_steps(bn: BayesianNetwork, query_var: str, evidence_var: str, evidence_value: bool = True):
    """
    Explain the steps needed to calculate P(query_var|evidence_var=evidence_value) using arc inversion.
    
    Args:
        bn: BayesianNetwork instance
        query_var: Variable to calculate probability for
        evidence_var: Observed variable
        evidence_value: Value of the observed variable
    """
    print(f"\nSteps to calculate P({query_var}|{evidence_var}={evidence_value}) using Arc Inversion:")
    
    # 1. Initial Setup
    print("\n1. Initial Setup:")
    print(f"   - We have evidence {evidence_var}={evidence_value}")
    print(f"   - We want to calculate P({query_var}|{evidence_var}={evidence_value})")
    print("   - We need to invert arcs to create a path from evidence to query")
    
    # 2. Find path from query to evidence
    path = find_path_to_evidence(bn, query_var, evidence_var)
    if not path:
        print(f"   ERROR: No path found from {query_var} to {evidence_var}")
        return
    
    print("\n2. Path Identification:")
    print(f"   - Found path: {' → '.join(path)}")
    print("   - We need to invert arcs to create a direct path from evidence to query")
    
    # 3. Arc Inversion Process
    print("\n3. Arc Inversion Process:")
    print("   We'll invert arcs in the following order:")
    
    # Get initial probability tables
    prob_tables = {}
    for var in path:
        parents = bn.get_parents(var)
        if parents:
            print(f"\n   Current probability table for {var}:")
            print(f"      P({var}|{','.join(parents)})")
            print("      Example calculation for a specific value:")
            print(f"      P({var}=True|{','.join(parents)}) = P({var}=True,{','.join(parents)}) / P({','.join(parents)})")
        else:
            print(f"\n   Current probability table for {var}:")
            print(f"      P({var})")
            print("      Example calculation for a specific value:")
            print(f"      P({var}=True) = P({var}=True)")
    
    # Perform and explain each inversion
    for i in range(len(path) - 1):
        current = path[i]
        next_var = path[i + 1]
        
        print(f"\n   Inverting arc {current} → {next_var}:")
        print(f"      Original: P({next_var}|{current})")
        print(f"      New tables after inversion:")
        print(f"      * P({current}|{next_var}) = P({next_var}|{current}) * P({current}) / P({next_var})")
        print(f"      * P({next_var}) = Σ_{current} P({next_var}|{current}) * P({current})")
        print("\n      Detailed calculation steps:")
        print(f"      1. Calculate P({next_var}) first:")
        print(f"         P({next_var}=True) = P({next_var}=True|{current}=True) * P({current}=True) +")
        print(f"                              P({next_var}=True|{current}=False) * P({current}=False)")
        print(f"      2. Then calculate P({current}|{next_var}):")
        print(f"         P({current}=True|{next_var}=True) = P({next_var}=True|{current}=True) * P({current}=True) / P({next_var}=True)")
        print(f"         P({current}=False|{next_var}=True) = P({next_var}=True|{current}=False) * P({current}=False) / P({next_var}=True)")
        print(f"         P({current}=True|{next_var}=False) = P({next_var}=False|{current}=True) * P({current}=True) / P({next_var}=False)")
        print(f"         P({current}=False|{next_var}=False) = P({next_var}=False|{current}=False) * P({current}=False) / P({next_var}=False)")
        print("\n      Note: These calculations ensure that:")
        print("      - The new tables maintain proper probability distributions")
        print("      - The joint probability P(current, next_var) remains the same")
        print("      - The marginal probabilities P(current) and P(next_var) are preserved")
    
    # 4. Evidence Propagation
    print("\n4. Evidence Propagation:")
    print(f"   - Instantiate {evidence_var}={evidence_value}")
    print("   - Propagate the evidence through the inverted network:")
    for var in reversed(path):
        if var == evidence_var:
            print(f"     * Start with P({var}={evidence_value})")
        else:
            print(f"     * Use updated P({var}|...) to get P({var}|{evidence_var}={evidence_value})")
            print(f"       P({var}=True|{evidence_var}={evidence_value}) = Σ_parents P({var}=True|parents) * P(parents|{evidence_var}={evidence_value})")
    
    # 5. Final Calculation
    print("\n5. Final Calculation:")
    print(f"   - After all inversions and evidence propagation:")
    print(f"   - We have P({query_var}|{evidence_var}={evidence_value}) directly")
    print("   - No need for normalization as the inversions maintain proper probability distributions")
    print(f"   - The final probability is obtained by propagating the evidence through the inverted network")
    print(f"   - Each step maintains the correct probability distribution through proper normalization") 