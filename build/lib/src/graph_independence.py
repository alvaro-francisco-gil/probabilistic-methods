import networkx as nx
from typing import List, Set, Tuple

class GraphIndependence:
    def __init__(self, nodes: List[str], edges: List[Tuple[str, str]]):
        """
        Initialize the graph with given nodes and edges.
        
        Args:
            nodes: List of node names
            edges: List of tuples representing edges between nodes
        """
        self.G = nx.Graph()
        self.G.add_nodes_from(nodes)
        self.G.add_edges_from(edges)

    def find_all_paths(self, start: str, end: str, observed: Set[str] = None) -> List[List[str]]:
        """
        Find all paths between two nodes, considering observed variables.
        
        Args:
            start: Starting node
            end: Ending node
            observed: Set of observed variables (default: None)
            
        Returns:
            List of all possible paths between start and end nodes
        """
        if observed is None:
            observed = set()
        
        all_paths = []
        visited = {start}
        
        def dfs(current: str, path: List[str]):
            if current == end:
                all_paths.append(path.copy())
                return
            
            for neighbor in self.G.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(neighbor, path)
                    path.pop()
                    visited.remove(neighbor)
        
        dfs(start, [start])
        return all_paths

    def is_path_active(self, path: List[str], observed: Set[str]) -> bool:
        """
        Check if a path is active given observed variables.
        
        Args:
            path: List of nodes representing a path
            observed: Set of observed variables
            
        Returns:
            True if the path is active, False if blocked
        """
        for i in range(1, len(path) - 1):
            prev, current, next_node = path[i-1], path[i], path[i+1]
            
            # Check for collider
            if prev in self.G.neighbors(current) and next_node in self.G.neighbors(current):
                # Path is blocked if collider is not in observed set
                if current not in observed:
                    return False
            # Check for non-collider
            else:
                # Path is blocked if non-collider is in observed set
                if current in observed:
                    return False
        return True

    def check_independence(self, X: str, Y: str, observed: Set[str] = None) -> Tuple[bool, List[Tuple[List[str], bool]]]:
        """
        Check if X is independent of Y given observed variables.
        
        Args:
            X: First variable
            Y: Second variable
            observed: Set of observed variables (default: None)
            
        Returns:
            Tuple containing:
            - Boolean indicating if X and Y are independent
            - List of tuples containing paths and their active status
        """
        if observed is None:
            observed = set()
        
        paths = self.find_all_paths(X, Y, observed)
        path_status = []
        
        for path in paths:
            is_active = self.is_path_active(path, observed)
            path_status.append((path, is_active))
        
        # If all paths are blocked, then X and Y are independent
        is_independent = all(not status for _, status in path_status)
        return is_independent, path_status

    def analyze_relationship(self, X: str, Y: str, observed: Set[str] = None) -> str:
        """
        Analyze and return a formatted string describing the independence relationship.
        
        Args:
            X: First variable
            Y: Second variable
            observed: Set of observed variables (default: None)
            
        Returns:
            Formatted string describing the independence relationship and paths
        """
        if observed is None:
            observed = set()
        
        is_independent, path_status = self.check_independence(X, Y, observed)
        
        result = f"IG({X}, {Y}"
        if observed:
            result += f"|{', '.join(sorted(observed))}"
        result += f"): {'True' if is_independent else 'False'}\n"
        
        result += "Paths:\n"
        for path, is_active in path_status:
            path_str = " -> ".join(path)
            result += f"- {path_str}: {'Active' if is_active else 'Blocked'}\n"
        
        return result

class DirectedGraphIndependence:
    def __init__(self, nodes: List[str], edges: List[Tuple[str, str]]):
        """
        Initialize the directed graph with given nodes and edges.
        
        Args:
            nodes: List of node names
            edges: List of tuples representing directed edges between nodes
        """
        self.G = nx.DiGraph()
        self.G.add_nodes_from(nodes)
        self.G.add_edges_from(edges)

    def find_all_paths(self, start: str, end: str, observed: Set[str] = None) -> List[List[str]]:
        """
        Find all possible paths between two nodes, considering both edge directions.
        
        Args:
            start: Starting node
            end: Ending node
            observed: Set of observed variables (default: None)
            
        Returns:
            List of all possible paths between start and end nodes
        """
        if observed is None:
            observed = set()
        
        all_paths = []
        visited = {start}
        
        def dfs(current: str, path: List[str]):
            if current == end:
                all_paths.append(path.copy())
                return
            
            # Check both successors and predecessors
            neighbors = set(self.G.successors(current)) | set(self.G.predecessors(current))
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(neighbor, path)
                    path.pop()
                    visited.remove(neighbor)
        
        dfs(start, [start])
        return all_paths

    def is_path_active(self, path: List[str], observed: Set[str]) -> bool:
        """
        Check if a path is active given observed variables.
        
        Args:
            path: List of nodes representing a path
            observed: Set of observed variables
            
        Returns:
            True if the path is active, False if blocked
        """
        for i in range(1, len(path) - 1):
            prev, current, next_node = path[i-1], path[i], path[i+1]
            
            # Check if it's a collider (v-structure)
            is_collider = ((prev, current) in self.G.edges and (next_node, current) in self.G.edges) or \
                         ((current, prev) in self.G.edges and (current, next_node) in self.G.edges)
            
            if is_collider:
                # Path is blocked if collider is not in observed set
                if current not in observed:
                    return False
            else:
                # Path is blocked if non-collider is in observed set
                if current in observed:
                    return False
        return True

    def check_independence(self, X: str, Y: str, observed: Set[str] = None) -> Tuple[bool, List[Tuple[List[str], bool]]]:
        """
        Check if X is independent of Y given observed variables in a directed graph.
        
        Args:
            X: First variable
            Y: Second variable
            observed: Set of observed variables (default: None)
            
        Returns:
            Tuple containing:
            - Boolean indicating if X and Y are independent
            - List of tuples containing paths and their active status
        """
        if observed is None:
            observed = set()
        
        paths = self.find_all_paths(X, Y, observed)
        path_status = []
        
        for path in paths:
            is_active = self.is_path_active(path, observed)
            path_status.append((path, is_active))
        
        # If all paths are blocked, then X and Y are independent
        is_independent = all(not status for _, status in path_status)
        return is_independent, path_status

    def analyze_relationship(self, X: str, Y: str, observed: Set[str] = None) -> str:
        """
        Analyze and return a formatted string describing the independence relationship.
        
        Args:
            X: First variable
            Y: Second variable
            observed: Set of observed variables (default: None)
            
        Returns:
            Formatted string describing the independence relationship and paths
        """
        if observed is None:
            observed = set()
        
        is_independent, path_status = self.check_independence(X, Y, observed)
        
        result = f"IG({X}, {Y}"
        if observed:
            result += f"|{', '.join(sorted(observed))}"
        result += f"): {'True' if is_independent else 'False'}\n"
        
        result += "Paths:\n"
        
        if not path_status:
            result += "• No paths found between nodes\n"
        else:
            for path, is_active in path_status:
                # Build path string with arrows
                path_parts = []
                for i in range(len(path) - 1):
                    current, next_node = path[i], path[i + 1]
                    if (current, next_node) in self.G.edges:
                        path_parts.append(f"{current} → {next_node}")
                    else:
                        path_parts.append(f"{current} ← {next_node}")
                
                # Construct the path string
                if not path_parts:
                    path_str = f"{X} → {Y}"  # Direct connection
                else:
                    # Get all nodes in order
                    nodes = [X]
                    for part in path_parts:
                        if " → " in part:
                            nodes.append(part.split(" → ")[1])
                        else:
                            nodes.append(part.split(" ← ")[1])
                    
                    # Build the path string with proper arrows
                    path_str = nodes[0]
                    for i in range(len(nodes) - 1):
                        current, next_node = nodes[i], nodes[i + 1]
                        if (current, next_node) in self.G.edges:
                            path_str += f" → {next_node}"
                        else:
                            path_str += f" ← {next_node}"
                
                # Add path status and reason
                result += f"• {path_str}: {'Active' if is_active else 'Blocked'}"
                
                # Add reason for the path status
                if not is_active:
                    # Find the blocking node
                    for i in range(1, len(path) - 1):
                        prev, current, next_node = path[i-1], path[i], path[i+1]
                        is_collider = ((prev, current) in self.G.edges and (next_node, current) in self.G.edges) or \
                                    ((current, prev) in self.G.edges and (current, next_node) in self.G.edges)
                        
                        if is_collider and current not in observed:
                            result += f" ({current} is a collider)"
                            break
                        elif not is_collider and current in observed:
                            result += f" ({current} is conditioned on)"
                            break
                else:
                    # Add reason for active path
                    for i in range(1, len(path) - 1):
                        prev, current, next_node = path[i-1], path[i], path[i+1]
                        is_collider = ((prev, current) in self.G.edges and (next_node, current) in self.G.edges) or \
                                    ((current, prev) in self.G.edges and (current, next_node) in self.G.edges)
                        
                        if is_collider and current in observed:
                            result += f" ({current} is an observed collider)"
                            break
                        elif not is_collider and current not in observed:
                            result += f" ({current} is not conditioned on)"
                            break
                
                result += "\n"
        
        # Add conclusion
        if not path_status:
            result += "Conclusion: No paths exist between nodes."
        elif is_independent:
            result += "Conclusion: All paths are blocked."
        else:
            result += "Conclusion: Active path exists."
        
        return result 