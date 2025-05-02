import numpy as np
import networkx as nx
from typing import Dict, List, Set, Tuple
from src.graph_independence import GraphIndependence, DirectedGraphIndependence

class ProbabilityDistribution:
    def __init__(self, probabilities: Dict[Tuple[bool, bool, bool], float]):
        """
        Initialize a probability distribution for three binary variables A, B, C.
        
        Args:
            probabilities: Dictionary mapping (a, b, c) tuples to their probabilities
        """
        self.probabilities = probabilities
        self.variables = ['A', 'B', 'C']
        
    def marginalize(self, variables: List[str]) -> Dict[Tuple[bool, ...], float]:
        """
        Marginalize the distribution over the given variables.
        
        Args:
            variables: List of variables to marginalize over
            
        Returns:
            Dictionary mapping tuples of variable values to their marginal probabilities
        """
        result = {}
        for (a, b, c), p in self.probabilities.items():
            key = tuple()
            if 'A' in variables:
                key += (a,)
            if 'B' in variables:
                key += (b,)
            if 'C' in variables:
                key += (c,)
            result[key] = result.get(key, 0) + p
        return result
    
    def conditional(self, X: str, Y: str, Z: str = None) -> Dict[Tuple[bool, bool], float]:
        """
        Calculate conditional probability P(X|Y) or P(X|Y,Z).
        
        Args:
            X: Target variable
            Y: First conditioning variable
            Z: Optional second conditioning variable
            
        Returns:
            Dictionary mapping (x, y) or (x, y, z) tuples to conditional probabilities
        """
        if Z is None:
            # P(X|Y)
            joint = self.marginalize([X, Y])
            marginal = self.marginalize([Y])
            result = {}
            for (x, y), p in joint.items():
                result[(x, y)] = p / marginal[(y,)]
            return result
        else:
            # P(X|Y,Z)
            joint = self.marginalize([X, Y, Z])
            marginal = self.marginalize([Y, Z])
            result = {}
            for (x, y, z), p in joint.items():
                result[(x, y, z)] = p / marginal[(y, z)]
            return result
    
    def check_independence(self, X: str, Y: str, Z: str = None) -> bool:
        """
        Check if X is independent of Y given Z.
        
        Args:
            X: First variable
            Y: Second variable
            Z: Optional conditioning variable
            
        Returns:
            True if X and Y are independent (given Z), False otherwise
        """
        if Z is None:
            # Check P(X,Y) = P(X)P(Y)
            joint = self.marginalize([X, Y])
            marginal_X = self.marginalize([X])
            marginal_Y = self.marginalize([Y])
            
            for (x, y), p_xy in joint.items():
                p_x = marginal_X[(x,)]
                p_y = marginal_Y[(y,)]
                if not np.isclose(p_xy, p_x * p_y, atol=1e-6):
                    return False
            return True
        else:
            # Check P(X,Y|Z) = P(X|Z)P(Y|Z)
            joint_cond = self.conditional(X, Z)
            joint_cond.update(self.conditional(Y, Z))
            joint_cond_XY = self.conditional(X, Y, Z)
            
            for (x, y, z), p_xy_z in joint_cond_XY.items():
                p_x_z = joint_cond.get((x, z), 0)
                p_y_z = joint_cond.get((y, z), 0)
                if not np.isclose(p_xy_z, p_x_z * p_y_z, atol=1e-6):
                    return False
            return True

def find_undirected_imaps(independence_relationships: Dict[str, bool]) -> List[Tuple[str, str]]:
    """
    Find all undirected graphs that are I-maps of the given independence relationships.
    
    Args:
        independence_relationships: Dictionary mapping independence statements to their truth values
        
    Returns:
        List of possible edge sets for undirected I-maps
    """
    # All possible edges between A, B, C
    possible_edges = [('A', 'B'), ('A', 'C'), ('B', 'C')]
    valid_imaps = []
    
    # Try all possible combinations of edges
    for i in range(2**3):  # 2^3 possible combinations
        edges = []
        for j, edge in enumerate(possible_edges):
            if (i >> j) & 1:  # Check if j-th bit is set
                edges.append(edge)
        
        # Create graph and check if it's an I-map
        graph = GraphIndependence(['A', 'B', 'C'], edges)
        is_imap = True
        
        # Check each independence relationship
        for rel, is_true in independence_relationships.items():
            # Parse the relationship
            if '|' in rel:
                # Conditional independence
                parts = rel.split('|')
                vars = parts[0].strip('IP()').split(',')
                X = vars[0].strip()
                Y = vars[1].strip()
                Z = parts[1].strip()
                indep = graph.check_independence(X, Y, {Z})[0]
            else:
                # Marginal independence
                vars = rel.strip('IP()').split(',')
                X = vars[0].strip()
                Y = vars[1].strip()
                indep = graph.check_independence(X, Y)[0]
            
            # If the graph implies independence when it shouldn't, it's not an I-map
            if indep and not is_true:
                is_imap = False
                break
        
        if is_imap:
            valid_imaps.append(edges)
    
    return valid_imaps

def find_directed_imaps(independence_relationships: Dict[str, bool]) -> List[Tuple[str, str]]:
    """
    Find all directed acyclic graphs that are I-maps of the given independence relationships.
    
    Args:
        independence_relationships: Dictionary mapping independence statements to their truth values
        
    Returns:
        List of possible edge sets for directed I-maps
    """
    # All possible directed edges between A, B, C
    possible_edges = [
        ('A', 'B'), ('B', 'A'),
        ('A', 'C'), ('C', 'A'),
        ('B', 'C'), ('C', 'B')
    ]
    valid_imaps = []
    
    # Try all possible combinations of edges (excluding cycles)
    for i in range(2**6):  # 2^6 possible combinations
        edges = []
        for j, edge in enumerate(possible_edges):
            if (i >> j) & 1:  # Check if j-th bit is set
                edges.append(edge)
        
        # Check if the graph is acyclic
        graph = DirectedGraphIndependence(['A', 'B', 'C'], edges)
        if not nx.is_directed_acyclic_graph(graph.G):
            continue
        
        # Check if it's an I-map
        is_imap = True
        for rel, is_true in independence_relationships.items():
            if '|' in rel:
                parts = rel.split('|')
                vars = parts[0].strip('IP()').split(',')
                X = vars[0].strip()
                Y = vars[1].strip()
                Z = parts[1].strip()
                indep = graph.check_independence(X, Y, {Z})[0]
            else:
                vars = rel.strip('IP()').split(',')
                X = vars[0].strip()
                Y = vars[1].strip()
                indep = graph.check_independence(X, Y)[0]
            
            if indep and not is_true:
                is_imap = False
                break
        
        if is_imap:
            valid_imaps.append(edges)
    
    return valid_imaps 