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
            for z_val in [True, False]:
                # Get P(Z=z)
                p_z = sum(prob for assignment, prob in self.probabilities.items()
                         if assignment[self.variables.index(Z)] == z_val)
                if p_z == 0:
                    continue
                
                for x_val in [True, False]:
                    for y_val in [True, False]:
                        # Calculate P(X=x, Y=y, Z=z)
                        p_xyz = sum(prob for assignment, prob in self.probabilities.items()
                                  if assignment[self.variables.index(X)] == x_val and
                                     assignment[self.variables.index(Y)] == y_val and
                                     assignment[self.variables.index(Z)] == z_val)
                        
                        # Calculate P(X=x, Z=z)
                        p_xz = sum(prob for assignment, prob in self.probabilities.items()
                                 if assignment[self.variables.index(X)] == x_val and
                                    assignment[self.variables.index(Z)] == z_val)
                        
                        # Calculate P(Y=y, Z=z)
                        p_yz = sum(prob for assignment, prob in self.probabilities.items()
                                 if assignment[self.variables.index(Y)] == y_val and
                                    assignment[self.variables.index(Z)] == z_val)
                        
                        # Calculate conditional probabilities
                        p_xy_given_z = p_xyz / p_z
                        p_x_given_z = p_xz / p_z
                        p_y_given_z = p_yz / p_z
                        
                        if not np.isclose(p_xy_given_z, p_x_given_z * p_y_given_z, atol=1e-6):
                            return False
            return True

    def print_independence_calculation(self, X: str, Y: str):
        """
        Print detailed calculations for marginal independence IP(X, Y).
        """
        idx_X = self.variables.index(X)
        idx_Y = self.variables.index(Y)
        print(f"\nCalculations for IP({X}, {Y}):")
        is_independent = True
        for x_val in [True, False]:
            for y_val in [True, False]:
                # Joint
                p_xy = sum(prob for assignment, prob in self.probabilities.items()
                           if assignment[idx_X] == x_val and assignment[idx_Y] == y_val)
                # Marginals
                p_x = sum(prob for assignment, prob in self.probabilities.items()
                          if assignment[idx_X] == x_val)
                p_y = sum(prob for assignment, prob in self.probabilities.items()
                          if assignment[idx_Y] == y_val)
                print(f"P({X}={x_val}, {Y}={y_val}) = {p_xy:.3f}, "
                      f"P({X}={x_val})*P({Y}={y_val}) = {p_x:.3f}*{p_y:.3f} = {(p_x*p_y):.3f}")
                if not np.isclose(p_xy, p_x * p_y, atol=1e-6):
                    is_independent = False
        if is_independent:
            print(f"\nConclusion: {X} and {Y} are independent.")
        else:
            print(f"\nConclusion: {X} and {Y} are NOT independent.")

    def print_conditional_independence_calculation(self, X: str, Y: str, Z: str):
        """
        Print detailed calculations for conditional independence IP(X, Y|Z).
        """
        idx_X = self.variables.index(X)
        idx_Y = self.variables.index(Y)
        idx_Z = self.variables.index(Z)
        print(f"\nCalculations for IP({X}, {Y}|{Z}):")
        is_independent = True
        for z_val in [True, False]:
            # Marginal for Z
            p_z = sum(prob for assignment, prob in self.probabilities.items()
                      if assignment[idx_Z] == z_val)
            if p_z == 0:
                continue
            for x_val in [True, False]:
                for y_val in [True, False]:
                    # Joint for X, Y, Z
                    p_xyz = sum(prob for assignment, prob in self.probabilities.items()
                                if assignment[idx_X] == x_val and assignment[idx_Y] == y_val and assignment[idx_Z] == z_val)
                    # P(X=x, Y=y | Z=z)
                    p_xy_given_z = p_xyz / p_z if p_z > 0 else 0

                    # P(X=x, Z=z)
                    p_xz = sum(prob for assignment, prob in self.probabilities.items()
                               if assignment[idx_X] == x_val and assignment[idx_Z] == z_val)
                    p_x_given_z = p_xz / p_z if p_z > 0 else 0

                    # P(Y=y, Z=z)
                    p_yz = sum(prob for assignment, prob in self.probabilities.items()
                               if assignment[idx_Y] == y_val and assignment[idx_Z] == z_val)
                    p_y_given_z = p_yz / p_z if p_z > 0 else 0

                    print(f"P({X}={x_val}, {Y}={y_val} | {Z}={z_val}) = {p_xy_given_z:.3f}, "
                          f"P({X}={x_val}|{Z}={z_val})*P({Y}={y_val}|{Z}={z_val}) = "
                          f"{p_x_given_z:.3f}*{p_y_given_z:.3f} = {(p_x_given_z*p_y_given_z):.3f}")
                    if not np.isclose(p_xy_given_z, p_x_given_z * p_y_given_z, atol=1e-6):
                        is_independent = False
        if is_independent:
            print(f"\nConclusion: {X} and {Y} are conditionally independent given {Z}.")
        else:
            print(f"\nConclusion: {X} and {Y} are NOT conditionally independent given {Z}.")

    def print_individual_probabilities(self):
        """
        Print all individual probabilities (marginals, joints, and conditionals) for better understanding.
        """
        print("\nIndividual Probabilities:")
        
        # Print marginals
        print("\nMarginal Probabilities:")
        for var in self.variables:
            idx = self.variables.index(var)
            p_true = sum(prob for assignment, prob in self.probabilities.items() if assignment[idx])
            p_false = 1 - p_true
            print(f"P({var}=True) = {p_true:.3f}")
            print(f"P({var}=False) = {p_false:.3f}")
        
        # Print joint probabilities for pairs
        print("\nJoint Probabilities (Pairs):")
        for i, var1 in enumerate(self.variables):
            for var2 in self.variables[i+1:]:
                idx1 = self.variables.index(var1)
                idx2 = self.variables.index(var2)
                print(f"\nP({var1}, {var2}):")
                for v1 in [True, False]:
                    for v2 in [True, False]:
                        p_joint = sum(prob for assignment, prob in self.probabilities.items()
                                    if assignment[idx1] == v1 and assignment[idx2] == v2)
                        print(f"P({var1}={v1}, {var2}={v2}) = {p_joint:.3f}")
        
        # Print conditional probabilities
        print("\nConditional Probabilities:")
        for var1 in self.variables:
            for var2 in self.variables:
                if var1 != var2:
                    idx1 = self.variables.index(var1)
                    idx2 = self.variables.index(var2)
                    print(f"\nP({var1} | {var2}):")
                    for v2 in [True, False]:
                        p_var2 = sum(prob for assignment, prob in self.probabilities.items()
                                   if assignment[idx2] == v2)
                        if p_var2 > 0:
                            for v1 in [True, False]:
                                p_joint = sum(prob for assignment, prob in self.probabilities.items()
                                            if assignment[idx1] == v1 and assignment[idx2] == v2)
                                p_cond = p_joint / p_var2
                                print(f"P({var1}={v1} | {var2}={v2}) = {p_cond:.3f}")

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

def plot_undirected_imaps(imaps: List[Tuple[str, str]], title: str = "Undirected I-maps"):
    """
    Plot all undirected I-maps in a single figure.
    
    Args:
        imaps: List of edge sets for undirected I-maps
        title: Title for the plot
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    import math
    
    # Calculate grid dimensions
    n_imaps = len(imaps)
    n_cols = min(3, n_imaps)  # Maximum 3 columns
    n_rows = math.ceil(n_imaps / n_cols)
    
    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot each I-map
    for i, edges in enumerate(imaps):
        G = nx.Graph()
        G.add_nodes_from(['A', 'B', 'C'])
        G.add_edges_from(edges)
        
        # Create subplot
        ax = axes[i]
        pos = nx.spring_layout(G, seed=42)  # Fixed seed for consistent layout
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                             node_size=1000, ax=ax)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=2, ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)
        
        # Set title for subplot
        ax.set_title(f"I-map {i+1}")
        ax.axis('off')
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    # Set main title
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

def plot_directed_imaps(imaps: List[Tuple[str, str]], title: str = "Directed Acyclic I-maps"):
    """
    Plot all directed acyclic I-maps in a single figure.
    
    Args:
        imaps: List of edge sets for directed I-maps
        title: Title for the plot
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    import math
    
    # Calculate grid dimensions
    n_imaps = len(imaps)
    n_cols = min(3, n_imaps)  # Maximum 3 columns
    n_rows = math.ceil(n_imaps / n_cols)
    
    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot each I-map
    for i, edges in enumerate(imaps):
        G = nx.DiGraph()
        G.add_nodes_from(['A', 'B', 'C'])
        G.add_edges_from(edges)
        
        # Create subplot
        ax = axes[i]
        
        # Use hierarchical layout for DAGs
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                             node_size=1000, ax=ax)
        
        # Draw edges with arrows
        nx.draw_networkx_edges(G, pos, width=2, 
                             arrows=True, arrowsize=20,
                             connectionstyle='arc3,rad=0.1',  # Curved edges
                             ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)
        
        # Set title for subplot
        ax.set_title(f"DAG I-map {i+1}")
        ax.axis('off')
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    # Set main title
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show() 