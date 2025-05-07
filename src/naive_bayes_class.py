import numpy as np
from typing import Dict, List, Set, Tuple

class NaiveBayesDiagnosis:
    def __init__(self):
        """Initialize the Naive Bayes diagnosis system."""
        # Prior probabilities
        self.priors = {
            'T': 0.001,  # P(T=True)
            'TB': 0.01   # P(TB=True)
        }
        
        # Conditional probabilities for symptoms given diseases
        # P(Symptom=True | Disease=True)
        self.symptom_given_disease = {
            'F': {  # Fever
                'T': 1.0,    # Typhoid always causes fever
                'TB': 0.6    # TB causes fever in 60% of cases
            },
            'B': {  # Bradycardia
                'T': 0.4,    # Typhoid causes bradycardia in 40% of cases
                'TB': 0.0    # TB doesn't cause bradycardia
            },
            'TC': {  # Tachycardia
                'T': 0.0,    # Typhoid doesn't cause tachycardia
                'TB': 0.58   # TB causes tachycardia in 58% of cases
            }
        }
        
        # Baseline probabilities (when no disease is present)
        self.baseline = {
            'F': 0.015,   # Fever baseline rate
            'B': 0.0005,  # Bradycardia baseline rate
            'TC': 0.013   # Tachycardia baseline rate
        }
    
    def calculate_symptom_probability(self, symptom: str, disease_values: Dict[str, bool]) -> float:
        """
        Calculate P(Symptom=True | Diseases) using the Naive Bayes assumption.
        
        Args:
            symptom: The symptom to calculate probability for
            disease_values: Dictionary mapping disease names to their values
            
        Returns:
            Probability of the symptom being True given the diseases
        """
        # If any disease that causes the symptom is present, use the disease probability
        for disease, is_present in disease_values.items():
            if is_present and disease in self.symptom_given_disease[symptom]:
                return self.symptom_given_disease[symptom][disease]
        
        # If no disease is present, use baseline probability
        return self.baseline[symptom]
    
    def calculate_disease_probability(self, disease: str, symptoms: Dict[str, bool], verbose: bool = False) -> float:
        """
        Calculate P(Disease=True | Symptoms) using Naive Bayes.
        
        Args:
            disease: The disease to calculate probability for
            symptoms: Dictionary mapping symptom names to their values
            verbose: Whether to print detailed calculations
            
        Returns:
            Probability of the disease being True given the symptoms
        """
        if verbose:
            print(f"\nCalculating P({disease}=True | {symptoms}):")
        
        # Calculate P(Symptoms | Disease=True)
        prob_symptoms_given_disease_true = 1.0
        for symptom, is_present in symptoms.items():
            if is_present:
                prob = self.calculate_symptom_probability(symptom, {disease: True})
                prob_symptoms_given_disease_true *= prob
                if verbose:
                    print(f"P({symptom}=True | {disease}=True) = {prob:.4f}")
            else:
                prob = 1 - self.calculate_symptom_probability(symptom, {disease: True})
                prob_symptoms_given_disease_true *= prob
                if verbose:
                    print(f"P({symptom}=False | {disease}=True) = {prob:.4f}")
        
        # Calculate P(Symptoms | Disease=False)
        prob_symptoms_given_disease_false = 1.0
        for symptom, is_present in symptoms.items():
            if is_present:
                prob = self.calculate_symptom_probability(symptom, {disease: False})
                prob_symptoms_given_disease_false *= prob
                if verbose:
                    print(f"P({symptom}=True | {disease}=False) = {prob:.4f}")
            else:
                prob = 1 - self.calculate_symptom_probability(symptom, {disease: False})
                prob_symptoms_given_disease_false *= prob
                if verbose:
                    print(f"P({symptom}=False | {disease}=False) = {prob:.4f}")
        
        # Calculate P(Disease=True | Symptoms) using Bayes' rule
        numerator = prob_symptoms_given_disease_true * self.priors[disease]
        denominator = (prob_symptoms_given_disease_true * self.priors[disease] +
                      prob_symptoms_given_disease_false * (1 - self.priors[disease]))
        
        posterior = numerator / denominator if denominator > 0 else 0
        
        if verbose:
            print(f"\nP({disease}=True | {symptoms}) = {posterior:.6f}")
        
        return posterior
    
    def diagnose(self, symptoms: Dict[str, bool], verbose: bool = False) -> Dict[str, float]:
        """
        Calculate probabilities for all diseases given the symptoms.
        
        Args:
            symptoms: Dictionary mapping symptom names to their values
            verbose: Whether to print detailed calculations
            
        Returns:
            Dictionary mapping disease names to their probabilities
        """
        if verbose:
            print("\nPerforming diagnosis using Naive Bayes:")
        
        diagnosis = {}
        for disease in ['T', 'TB']:
            prob = self.calculate_disease_probability(disease, symptoms, verbose)
            diagnosis[disease] = prob
            print(f"\nP({disease}=True | {symptoms}) = {prob:.6f}")
            print(f"P({disease}=False | {symptoms}) = {1-prob:.6f}")
        
        return diagnosis 