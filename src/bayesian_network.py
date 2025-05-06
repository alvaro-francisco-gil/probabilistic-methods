import numpy as np
from typing import Dict, List, Set, Tuple, Optional
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

class ConditionalProbabilityTable:
    def __init__(self, variable: str, parents: List[str]):
        """
        Initialize a Conditional Probability Table (CPT).
        
        Args:
            variable: The name of the variable this CPT represents
            parents: List of parent variable names
        """
        self.variable = variable
        self.parents = parents
        self.table = {}  # Maps (variable_value, parent_values) to probability
        
    def add_probability(self, variable_value: bool, parent_values: Dict[str, bool], probability: float):
        """
        Add a probability to the CPT.
        
        Args:
            variable_value: Value of the variable (True/False)
            parent_values: Dictionary mapping parent names to their values
            probability: The probability value
        """
        # Create a tuple of parent values in the correct order
        parent_tuple = tuple(parent_values[parent] for parent in self.parents)
        self.table[(variable_value, parent_tuple)] = probability
        
    def get_probability(self, variable_value: bool, parent_values: Dict[str, bool]) -> float:
        """
        Get a probability from the CPT.
        
        Args:
            variable_value: Value of the variable (True/False)
            parent_values: Dictionary mapping parent names to their values
            
        Returns:
            The probability value
        """
        parent_tuple = tuple(parent_values[parent] for parent in self.parents)
        return self.table.get((variable_value, parent_tuple), 0.0)
    
    def print_table(self, verbose: bool = False):
        """
        Print the CPT in a readable format.
        
        Args:
            verbose: If True, print all probabilities. If False, print only non-zero probabilities.
        """
        if not verbose:
            print(f"\nCPT for {self.variable}:")
            if not self.parents:
                for var_val in [True, False]:
                    prob = self.table.get((var_val, ()), 0.0)
                    if prob > 0 or verbose:
                        print(f"P({self.variable}={var_val}) = {prob:.4f}")
            else:
                # Print only non-zero probabilities
                for var_val in [True, False]:
                    for parent_combo in self._generate_parent_combinations():
                        parent_dict = dict(zip(self.parents, parent_combo))
                        prob = self.get_probability(var_val, parent_dict)
                        if prob > 0 or verbose:
                            parent_str = ", ".join(f"{p}={v}" for p, v in parent_dict.items())
                            print(f"P({self.variable}={var_val} | {parent_str}) = {prob:.4f}")
            return

        # Verbose printing
        print(f"\nConditional Probability Table for {self.variable}")
        if not self.parents:
            print("No parents (prior probability)")
            for var_val in [True, False]:
                prob = self.table.get((var_val, ()), 0.0)
                print(f"P({self.variable}={var_val}) = {prob:.4f}")
        else:
            # Print header
            header = " | ".join(self.parents + [self.variable, "Probability"])
            print(header)
            print("-" * len(header))
            
            # Print each row
            for var_val in [True, False]:
                for parent_combo in self._generate_parent_combinations():
                    parent_dict = dict(zip(self.parents, parent_combo))
                    prob = self.get_probability(var_val, parent_dict)
                    row = " | ".join(str(val) for val in list(parent_combo) + [var_val, f"{prob:.4f}"])
                    print(row)
    
    def _generate_parent_combinations(self) -> List[Tuple[bool, ...]]:
        """Generate all possible combinations of parent values."""
        n_parents = len(self.parents)
        return [tuple(bool((i >> j) & 1) for j in range(n_parents))
                for i in range(2**n_parents)]

class BayesianNetwork:
    def __init__(self):
        """Initialize an empty Bayesian network."""
        self.G = nx.DiGraph()
        self.cpts = {}  # Maps variable names to their CPTs
        self.variables = set()
        
    def add_variable(self, variable: str):
        """Add a variable to the network."""
        self.variables.add(variable)
        self.G.add_node(variable)
        
    def add_edge(self, parent: str, child: str):
        """Add a directed edge from parent to child."""
        self.G.add_edge(parent, child)
        
    def add_cpt(self, cpt: ConditionalProbabilityTable):
        """Add a CPT to the network."""
        self.cpts[cpt.variable] = cpt
        
    def get_parents(self, variable: str) -> List[str]:
        """Get the parents of a variable."""
        return list(self.G.predecessors(variable))
    
    def get_children(self, variable: str) -> List[str]:
        """Get the children of a variable."""
        return list(self.G.successors(variable))
    
    def plot_network(self, title: str = "Bayesian Network"):
        """Plot the network structure using matplotlib."""
        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(self.G, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.G, pos, node_color='lightblue', 
                             node_size=2000, alpha=0.7)
        
        # Draw edges with arrows
        nx.draw_networkx_edges(self.G, pos, width=2, 
                             arrows=True, arrowsize=20,
                             connectionstyle='arc3,rad=0.1')
        
        # Draw labels
        nx.draw_networkx_labels(self.G, pos, font_size=12, font_weight='bold')
        
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def print_network(self, verbose: bool = False):
        """
        Print the network structure and CPTs.
        
        Args:
            verbose: If True, print all probabilities. If False, print only non-zero probabilities.
        """
        print("\nBayesian Network Structure:")
        for node in self.G.nodes():
            parents = self.get_parents(node)
            if parents:
                print(f"{node} <- {', '.join(parents)}")
            else:
                print(f"{node} (no parents)")
        
        print("\nConditional Probability Tables:")
        for variable in self.variables:
            if variable in self.cpts:
                self.cpts[variable].print_table(verbose)
    
    def _get_missing_parent_values(self, variable: str, evidence: Dict[str, bool]) -> List[Dict[str, bool]]:
        """
        Get all possible combinations of values for missing parents.
        
        Args:
            variable: The variable whose parents we're considering
            evidence: Current evidence dictionary
            
        Returns:
            List of dictionaries with all possible combinations of missing parent values
        """
        parents = self.get_parents(variable)
        missing_parents = [p for p in parents if p not in evidence]
        
        if not missing_parents:
            return [{}]
        
        # Generate all possible combinations for missing parents
        n_missing = len(missing_parents)
        combinations = []
        for i in range(2**n_missing):
            values = [bool((i >> j) & 1) for j in range(n_missing)]
            parent_dict = dict(zip(missing_parents, values))
            combinations.append(parent_dict)
        
        return combinations
    
    def calculate_joint_probability(self, evidence: Dict[str, bool], verbose: bool = False) -> float:
        """
        Calculate the joint probability of a specific configuration.
        
        Args:
            evidence: Dictionary mapping variable names to their values
            verbose: If True, print detailed calculations
            
        Returns:
            The joint probability
        """
        if verbose:
            print("\nCalculating joint probability:")
        joint_prob = 1.0
        
        # Process variables in topological order
        for variable in nx.topological_sort(self.G):
            if variable in evidence:
                var_value = evidence[variable]
                parents = self.get_parents(variable)
                
                # Get all possible combinations of missing parent values
                missing_combinations = self._get_missing_parent_values(variable, evidence)
                
                # Sum over all possible combinations of missing parent values
                var_prob = 0.0
                for missing_values in missing_combinations:
                    # Combine known and missing parent values
                    parent_values = {p: evidence[p] for p in parents if p in evidence}
                    parent_values.update(missing_values)
                    
                    # Get probability from CPT
                    prob = self.cpts[variable].get_probability(var_value, parent_values)
                    var_prob += prob
                
                joint_prob *= var_prob
                
                if verbose:
                    # Print calculation
                    parent_str = ", ".join(f"{p}={v}" for p, v in parent_values.items())
                    if parent_str:
                        print(f"P({variable}={var_value} | {parent_str}) = {var_prob:.4f}")
                    else:
                        print(f"P({variable}={var_value}) = {var_prob:.4f}")
        
        if verbose:
            print(f"\nFinal joint probability: {joint_prob:.6f}")
        return joint_prob
    
    def calculate_conditional_probability(self, query: Dict[str, bool], 
                                       evidence: Dict[str, bool],
                                       verbose: bool = False) -> float:
        """
        Calculate conditional probability using the chain rule.
        
        Args:
            query: Dictionary mapping query variables to their values
            evidence: Dictionary mapping evidence variables to their values
            verbose: If True, print detailed calculations
            
        Returns:
            The conditional probability
        """
        if verbose:
            print(f"\nCalculating P({query} | {evidence}):")
        
        # Combine query and evidence
        full_evidence = {**evidence, **query}
        
        # Calculate joint probability of query and evidence
        joint_prob = self.calculate_joint_probability(full_evidence, verbose)
        
        # Calculate probability of evidence
        evidence_prob = self.calculate_joint_probability(evidence, verbose)
        
        # Calculate conditional probability
        cond_prob = joint_prob / evidence_prob if evidence_prob > 0 else 0
        
        if verbose:
            print(f"\nP({query} | {evidence}) = {cond_prob:.6f}")
        return cond_prob
    
    def diagnose(self, symptoms: Dict[str, bool], verbose: bool = False) -> Dict[str, float]:
        """
        Calculate the probability of each disease given the symptoms.
        
        Args:
            symptoms: Dictionary mapping symptom variables to their values
            verbose: If True, print detailed calculations
            
        Returns:
            Dictionary mapping disease names to their probabilities
        """
        if verbose:
            print("\nPerforming diagnosis:")
        diagnosis = {}
        
        # Get all disease variables (nodes with no parents)
        diseases = [node for node in self.G.nodes() if not self.get_parents(node)]
        
        for disease in diseases:
            # Calculate P(disease=True | symptoms)
            prob_true = self.calculate_conditional_probability(
                {disease: True}, symptoms, verbose)
            
            # Calculate P(disease=False | symptoms)
            prob_false = self.calculate_conditional_probability(
                {disease: False}, symptoms, verbose)
            
            # Normalize probabilities
            total = prob_true + prob_false
            if total > 0:
                prob_true /= total
                prob_false /= total
            
            diagnosis[disease] = prob_true
            print(f"\nP({disease}=True | {symptoms}) = {prob_true:.6f}")
            print(f"P({disease}=False | {symptoms}) = {prob_false:.6f}")
        
        return diagnosis 