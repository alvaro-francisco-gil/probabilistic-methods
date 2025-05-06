import numpy as np
from typing import Dict, List, Set, Tuple, Optional
import networkx as nx
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
    
    def print_table(self):
        """Print the CPT in a readable format."""
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
                    row = " | ".join(str(val) for val in parent_combo + [var_val, f"{prob:.4f}"])
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
    
    def print_network(self):
        """Print the network structure and all CPTs."""
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
                self.cpts[variable].print_table()
    
    def calculate_joint_probability(self, evidence: Dict[str, bool]) -> float:
        """
        Calculate the joint probability of a specific configuration.
        
        Args:
            evidence: Dictionary mapping variable names to their values
            
        Returns:
            The joint probability
        """
        print("\nCalculating joint probability:")
        joint_prob = 1.0
        
        # Process variables in topological order
        for variable in nx.topological_sort(self.G):
            if variable in evidence:
                var_value = evidence[variable]
                parent_values = {p: evidence[p] for p in self.get_parents(variable)}
                
                # Get probability from CPT
                prob = self.cpts[variable].get_probability(var_value, parent_values)
                joint_prob *= prob
                
                # Print calculation
                parent_str = ", ".join(f"{p}={v}" for p, v in parent_values.items())
                if parent_str:
                    print(f"P({variable}={var_value} | {parent_str}) = {prob:.4f}")
                else:
                    print(f"P({variable}={var_value}) = {prob:.4f}")
        
        print(f"\nFinal joint probability: {joint_prob:.6f}")
        return joint_prob
    
    def calculate_conditional_probability(self, query: Dict[str, bool], 
                                       evidence: Dict[str, bool]) -> float:
        """
        Calculate conditional probability using the chain rule.
        
        Args:
            query: Dictionary mapping query variables to their values
            evidence: Dictionary mapping evidence variables to their values
            
        Returns:
            The conditional probability
        """
        print(f"\nCalculating P({query} | {evidence}):")
        
        # Combine query and evidence
        full_evidence = {**evidence, **query}
        
        # Calculate joint probability of query and evidence
        joint_prob = self.calculate_joint_probability(full_evidence)
        
        # Calculate probability of evidence
        evidence_prob = self.calculate_joint_probability(evidence)
        
        # Calculate conditional probability
        cond_prob = joint_prob / evidence_prob if evidence_prob > 0 else 0
        
        print(f"\nP({query} | {evidence}) = {cond_prob:.6f}")
        return cond_prob
    
    def diagnose(self, symptoms: Dict[str, bool]) -> Dict[str, float]:
        """
        Calculate the probability of each disease given the symptoms.
        
        Args:
            symptoms: Dictionary mapping symptom variables to their values
            
        Returns:
            Dictionary mapping disease names to their probabilities
        """
        print("\nPerforming diagnosis:")
        diagnosis = {}
        
        # Get all disease variables (nodes with no parents)
        diseases = [node for node in self.G.nodes() if not self.get_parents(node)]
        
        for disease in diseases:
            # Calculate P(disease=True | symptoms)
            prob_true = self.calculate_conditional_probability(
                {disease: True}, symptoms)
            
            # Calculate P(disease=False | symptoms)
            prob_false = self.calculate_conditional_probability(
                {disease: False}, symptoms)
            
            # Normalize probabilities
            total = prob_true + prob_false
            if total > 0:
                prob_true /= total
                prob_false /= total
            
            diagnosis[disease] = prob_true
            print(f"\nP({disease}=True | {symptoms}) = {prob_true:.6f}")
            print(f"P({disease}=False | {symptoms}) = {prob_false:.6f}")
        
        return diagnosis 