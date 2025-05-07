from bayesian_network import BayesianNetwork, ConditionalProbabilityTable

def create_network():
    # Create the network
    bn = BayesianNetwork()
    
    # Add variables
    variables = ['A', 'B', 'C', 'D', 'F', 'G']
    for var in variables:
        bn.add_variable(var)
    
    # Add edges
    edges = [
        ('A', 'B'),
        ('B', 'C'),
        ('B', 'D'),
        ('C', 'F'),
        ('D', 'F'),
        ('D', 'G')
    ]
    for parent, child in edges:
        bn.add_edge(parent, child)
    
    # Add CPTs
    # P(A)
    cpt_a = ConditionalProbabilityTable('A', [])
    cpt_a.add_probability(True, {}, 0.1)    # P(+a) = 0.1
    cpt_a.add_probability(False, {}, 0.9)   # P(¬a) = 0.9
    bn.add_cpt(cpt_a)
    
    # P(B|A)
    cpt_b = ConditionalProbabilityTable('B', ['A'])
    cpt_b.add_probability(True, {'A': True}, 0.8)    # P(+b|+a) = 0.8
    cpt_b.add_probability(False, {'A': True}, 0.2)   # P(¬b|+a) = 0.2
    cpt_b.add_probability(True, {'A': False}, 0.25)  # P(+b|¬a) = 0.25
    cpt_b.add_probability(False, {'A': False}, 0.75) # P(¬b|¬a) = 0.75
    bn.add_cpt(cpt_b)
    
    # P(C|B)
    cpt_c = ConditionalProbabilityTable('C', ['B'])
    cpt_c.add_probability(True, {'B': True}, 0.7)    # P(+c|+b) = 0.7
    cpt_c.add_probability(False, {'B': True}, 0.3)   # P(¬c|+b) = 0.3
    cpt_c.add_probability(True, {'B': False}, 0.35)  # P(+c|¬b) = 0.35
    cpt_c.add_probability(False, {'B': False}, 0.65) # P(¬c|¬b) = 0.65
    bn.add_cpt(cpt_c)
    
    # P(D|B)
    cpt_d = ConditionalProbabilityTable('D', ['B'])
    cpt_d.add_probability(True, {'B': True}, 0.6)    # P(+d|+b) = 0.6
    cpt_d.add_probability(False, {'B': True}, 0.4)   # P(¬d|+b) = 0.4
    cpt_d.add_probability(True, {'B': False}, 0.1)   # P(+d|¬b) = 0.1
    cpt_d.add_probability(False, {'B': False}, 0.9)  # P(¬d|¬b) = 0.9
    bn.add_cpt(cpt_d)
    
    # P(F|C,D)
    cpt_f = ConditionalProbabilityTable('F', ['C', 'D'])
    cpt_f.add_probability(True, {'C': True, 'D': True}, 0.8)     # P(+f|+c,+d) = 0.8
    cpt_f.add_probability(False, {'C': True, 'D': True}, 0.2)    # P(¬f|+c,+d) = 0.2
    cpt_f.add_probability(True, {'C': True, 'D': False}, 0.6)    # P(+f|+c,¬d) = 0.6
    cpt_f.add_probability(False, {'C': True, 'D': False}, 0.4)   # P(¬f|+c,¬d) = 0.4
    cpt_f.add_probability(True, {'C': False, 'D': True}, 0.5)    # P(+f|¬c,+d) = 0.5
    cpt_f.add_probability(False, {'C': False, 'D': True}, 0.5)   # P(¬f|¬c,+d) = 0.5
    cpt_f.add_probability(True, {'C': False, 'D': False}, 0.0)   # P(+f|¬c,¬d) = 0.0
    cpt_f.add_probability(False, {'C': False, 'D': False}, 1.0)  # P(¬f|¬c,¬d) = 1.0
    bn.add_cpt(cpt_f)
    
    # P(G|D)
    cpt_g = ConditionalProbabilityTable('G', ['D'])
    cpt_g.add_probability(True, {'D': True}, 0.4)    # P(+g|+d) = 0.4
    cpt_g.add_probability(False, {'D': True}, 0.6)   # P(¬g|+d) = 0.6
    cpt_g.add_probability(True, {'D': False}, 0.1)   # P(+g|¬d) = 0.1
    cpt_g.add_probability(False, {'D': False}, 0.9)  # P(¬g|¬d) = 0.9
    bn.add_cpt(cpt_g)
    
    return bn

def main():
    # Create the network
    bn = create_network()
    
    # Print network structure and CPTs
    print("Bayesian Network Structure and CPTs:")
    bn.print_network(verbose=True)
    
    # Calculate P(b|¬a, +f, ¬g)
    query = {'B': True}
    evidence = {'A': False, 'F': True, 'G': False}
    
    print("\nCalculating P(b|¬a, +f, ¬g) using variable elimination:")
    prob = bn.variable_elimination(query, evidence, verbose=True)
    print(f"\nFinal result: P(b|¬a, +f, ¬g) = {prob:.6f}")

if __name__ == "__main__":
    main() 