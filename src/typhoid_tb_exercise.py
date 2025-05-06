from bayesian_network import BayesianNetwork, ConditionalProbabilityTable

def create_typhoid_tb_network():
    """Create the Bayesian network for the typhoid and TB exercise."""
    # Create network
    bn = BayesianNetwork()
    
    # Add variables
    variables = ['T', 'TB', 'F', 'B', 'TC']  # Typhoid, Tuberculosis, Fever, Bradycardia, Tachycardia
    for var in variables:
        bn.add_variable(var)
    
    # Add edges
    bn.add_edge('T', 'F')  # Typhoid causes Fever
    bn.add_edge('T', 'B')  # Typhoid causes Bradycardia
    bn.add_edge('TB', 'F')  # TB causes Fever
    bn.add_edge('TB', 'TC')  # TB causes Tachycardia
    
    # Create CPTs
    
    # Prior probabilities for diseases
    t_cpt = ConditionalProbabilityTable('T', [])
    t_cpt.add_probability(True, {}, 0.001)  # P(T=True) = 0.001
    t_cpt.add_probability(False, {}, 0.999)  # P(T=False) = 0.999
    
    tb_cpt = ConditionalProbabilityTable('TB', [])
    tb_cpt.add_probability(True, {}, 0.01)  # P(TB=True) = 0.01
    tb_cpt.add_probability(False, {}, 0.99)  # P(TB=False) = 0.99
    
    # Fever CPT (OR gate for T and TB)
    f_cpt = ConditionalProbabilityTable('F', ['T', 'TB'])
    # P(F=True | T=True, TB=True) = 1 (if either disease is present, fever is certain)
    f_cpt.add_probability(True, {'T': True, 'TB': True}, 1.0)
    f_cpt.add_probability(False, {'T': True, 'TB': True}, 0.0)
    # P(F=True | T=True, TB=False) = 1 (typhoid always causes fever)
    f_cpt.add_probability(True, {'T': True, 'TB': False}, 1.0)
    f_cpt.add_probability(False, {'T': True, 'TB': False}, 0.0)
    # P(F=True | T=False, TB=True) = 0.6 (TB causes fever in 60% of cases)
    f_cpt.add_probability(True, {'T': False, 'TB': True}, 0.6)
    f_cpt.add_probability(False, {'T': False, 'TB': True}, 0.4)
    # P(F=True | T=False, TB=False) = 0.015 (baseline fever rate)
    f_cpt.add_probability(True, {'T': False, 'TB': False}, 0.015)
    f_cpt.add_probability(False, {'T': False, 'TB': False}, 0.985)
    
    # Bradycardia CPT
    b_cpt = ConditionalProbabilityTable('B', ['T'])
    # P(B=True | T=True) = 0.4 (typhoid causes bradycardia in 40% of cases)
    b_cpt.add_probability(True, {'T': True}, 0.4)
    b_cpt.add_probability(False, {'T': True}, 0.6)
    # P(B=True | T=False) = 0.0005 (baseline bradycardia rate)
    b_cpt.add_probability(True, {'T': False}, 0.0005)
    b_cpt.add_probability(False, {'T': False}, 0.9995)
    
    # Tachycardia CPT
    tc_cpt = ConditionalProbabilityTable('TC', ['TB'])
    # P(TC=True | TB=True) = 0.58 (TB causes tachycardia in 58% of cases)
    tc_cpt.add_probability(True, {'TB': True}, 0.58)
    tc_cpt.add_probability(False, {'TB': True}, 0.42)
    # P(TC=True | TB=False) = 0.013 (baseline tachycardia rate)
    tc_cpt.add_probability(True, {'TB': False}, 0.013)
    tc_cpt.add_probability(False, {'TB': False}, 0.987)
    
    # Add CPTs to network
    bn.add_cpt(t_cpt)
    bn.add_cpt(tb_cpt)
    bn.add_cpt(f_cpt)
    bn.add_cpt(b_cpt)
    bn.add_cpt(tc_cpt)
    
    return bn

def main():
    # Create the network
    bn = create_typhoid_tb_network()
    
    print("Exercise 1.4 - Typhoid and Tuberculosis Bayesian Network")
    
    # 1. Network Structure and CPTs
    print("\n1. Network Structure and CPTs:")
    bn.plot_network("Typhoid and TB Bayesian Network")
    bn.print_network(verbose=False)  # Only show non-zero probabilities
    
    # 2. Variable values
    print("\n2. Variable Values:")
    print("All variables are binary (True/False)")
    print("T: Typhoid")
    print("TB: Tuberculosis")
    print("F: Fever")
    print("B: Bradycardia")
    print("TC: Tachycardia")
    
    # 3. Conditional probabilities are shown in the CPTs above
    
    # 4. Hypotheses
    print("\n4. Hypotheses:")
    print("1. Diseases are independent (no direct causal relationship between T and TB)")
    print("2. Symptoms are conditionally independent given their causes")
    print("3. Fever follows an OR gate when both diseases are present")
    print("4. Baseline rates apply when no disease is present")
    print("\nThese hypotheses are reasonable because:")
    print("- Diseases are independent in the general population")
    print("- Symptoms are direct effects of the diseases")
    print("- Fever is a common symptom that can be caused by either disease")
    print("- Baseline rates represent the general population without either disease")
    
    # 5. Diagnosis table
    print("\n5. Diagnosis Table (showing two detailed calculations):")
    
    # Example 1: Fever and Tachycardia
    print("\nExample 1: Patient has Fever and Tachycardia")
    symptoms = {'F': True, 'TC': True}
    diagnosis = bn.diagnose(symptoms, verbose=True)  # Show detailed calculations
    
    # Example 2: Fever and Bradycardia
    print("\nExample 2: Patient has Fever and Bradycardia")
    symptoms = {'F': True, 'B': True}
    diagnosis = bn.diagnose(symptoms, verbose=True)  # Show detailed calculations
    
    # 6. Association between Fever and Tachycardia in TB
    print("\n6. Association between Fever and Tachycardia in TB:")
    print("In our model, Fever and Tachycardia are conditionally independent given TB.")
    print("This means that in TB patients, the presence of fever doesn't directly affect")
    print("the probability of tachycardia. However, in reality, there might be a")
    print("physiological connection between these symptoms in TB patients.")
    print("\nTo model this association, we would need to:")
    print("1. Add a direct edge between F and TC")
    print("2. Create a new CPT for TC that depends on both TB and F")
    print("3. Adjust the probabilities to reflect the observed association")
    print("\nThis would make the model more accurate but also more complex.")

if __name__ == "__main__":
    main() 