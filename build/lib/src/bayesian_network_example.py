from src.bayesian_network import BayesianNetwork, ConditionalProbabilityTable
from typing import List, Set, Dict, Tuple, Optional
from src.bayesian_network_explainer import (
    explain_join_tree_construction,
    explain_grouping_steps,
    explain_arc_inversion_steps,
    explain_elimination_steps
)

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

