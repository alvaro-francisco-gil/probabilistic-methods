import numpy as np
from typing import Dict, List, Set, Tuple
import matplotlib.pyplot as plt

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

def main():
    # Create the Naive Bayes diagnosis system
    nb = NaiveBayesDiagnosis()
    
    print("Exercise 1.5 - Typhoid and Tuberculosis using Naive Bayes")
    
    # 1. Model Structure
    print("\n1. Model Structure:")
    print("In Naive Bayes, we assume all symptoms are conditionally independent given the diseases.")
    print("The structure is simpler than a Bayesian network - we only need:")
    print("- Prior probabilities for each disease")
    print("- Conditional probabilities for each symptom given each disease")
    print("- Baseline probabilities for symptoms when no disease is present")
    
    # 2. Variable values
    print("\n2. Variable Values:")
    print("All variables are binary (True/False)")
    print("T: Typhoid")
    print("TB: Tuberculosis")
    print("F: Fever")
    print("B: Bradycardia")
    print("TC: Tachycardia")
    
    # 3. Conditional probabilities
    print("\n3. Conditional Probabilities:")
    print("Prior probabilities:")
    for disease, prob in nb.priors.items():
        print(f"P({disease}=True) = {prob:.4f}")
    
    print("\nConditional probabilities P(Symptom=True | Disease=True):")
    for symptom in ['F', 'B', 'TC']:
        print(f"\n{symptom} (Fever/Bradycardia/Tachycardia):")
        for disease in ['T', 'TB']:
            prob = nb.symptom_given_disease[symptom].get(disease, 0.0)
            print(f"P({symptom}=True | {disease}=True) = {prob:.4f}")
    
    print("\nBaseline probabilities (no disease):")
    for symptom, prob in nb.baseline.items():
        print(f"P({symptom}=True) = {prob:.4f}")
    
    # 4. Hypotheses
    print("\n4. Hypotheses:")
    print("1. Diseases are independent (same as Bayesian network)")
    print("2. Symptoms are conditionally independent given the diseases (Naive Bayes assumption)")
    print("3. Each symptom's probability depends only on the presence/absence of diseases")
    print("4. Baseline rates apply when no disease is present")
    print("\nThese hypotheses are different from the Bayesian network because:")
    print("- We don't model causal relationships between symptoms")
    print("- We don't use an OR gate for fever - each disease contributes independently")
    print("- The model is simpler but less accurate in representing real-world relationships")
    
    # 5. Diagnosis table
    print("\n5. Diagnosis Table (showing two detailed calculations):")
    
    # Example 1: Fever and Tachycardia
    print("\nExample 1: Patient has Fever and Tachycardia")
    symptoms = {'F': True, 'TC': True}
    diagnosis = nb.diagnose(symptoms, verbose=True)
    
    # Example 2: Fever and Bradycardia
    print("\nExample 2: Patient has Fever and Bradycardia")
    symptoms = {'F': True, 'B': True}
    diagnosis = nb.diagnose(symptoms, verbose=True)
    
    # 6. Association between Fever and Tachycardia in TB
    print("\n6. Association between Fever and Tachycardia in TB:")
    print("In the Naive Bayes model, Fever and Tachycardia are conditionally independent")
    print("given TB, just like in the Bayesian network. However, the way we calculate")
    print("probabilities is different:")
    print("1. In Naive Bayes, we multiply the individual probabilities")
    print("2. In the Bayesian network, we use the actual causal relationships")
    print("\nThe Naive Bayes model is simpler but less accurate because it doesn't")
    print("capture the physiological connection between these symptoms in TB patients.")

if __name__ == "__main__":
    main() 