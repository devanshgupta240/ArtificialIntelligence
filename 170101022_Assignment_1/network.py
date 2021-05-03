from pomegranate import *

travel = DiscreteDistribution({'T': 0.05, 'F': 0.95})
ownsDevice = DiscreteDistribution({'T': 0.7, 'F': 0.3})




onlinePurchase = ConditionalProbabilityTable(
        [['T', 'T', 0.4],
         ['T', 'F', 0.6],
         ['F', 'T', 0.05],
         ['F', 'F', 0.95]], [ownsDevice])

foreignPurchacse = ConditionalProbabilityTable(
        [['T', 'T', 0.88],
         ['T', 'F', 0.12],
         ['F', 'T', 0.0001],
         ['F', 'F', 0.9999]], [travel])

fraud = ConditionalProbabilityTable(
        [['T', 'T', 'T', 0.995],
         ['T', 'T', 'F', 0.005],
         ['T', 'F', 'T', 0.85],
         ['T', 'F', 'F', 0.15],
         ['F', 'T', 'T', 0.8],
         ['F', 'T', 'F', 0.2],
         ['F', 'F', 'T', 0.75],
         ['F', 'F', 'F', 0.25]], [travel, onlinePurchase])

s1 = Node(travel, name="travel")
s2 = Node(ownsDevice, name="ownsDevice")
s3 = Node(onlinePurchase, name="onlinePurchase")
s4 = Node(foreignPurchacse, name="foreignPurchacse")
s5 = Node(fraud,name="fraud")



model = BayesianNetwork("Fraudelent Detection")
model.add_states(s1, s2, s3, s4, s5)
model.add_edge(s1, s4)
model.add_edge(s1, s5)
model.add_edge(s2, s3)
model.add_edge(s3, s5)
model.bake()




beliefs = model.predict_proba({})
beliefs = map(str,beliefs)

for state, belief in zip(model.states, beliefs):
        if state.name == 'fraud':
                print("P(fraud)")
                print(belief)

print("\n")
beliefs = model.predict_proba({'ownsDevice': 'T'})
beliefs = map(str,beliefs)

for state, belief in zip(model.states, beliefs):
        if state.name == 'fraud':
                #print(state.name)
                print("P(fraud|ownsDevice=True)")
                print(belief)

print("\n")
beliefs = model.predict_proba({'travel': 'T'})
beliefs = map(str,beliefs)

for state, belief in zip(model.states, beliefs):
        if state.name == 'fraud':
                print("P(fraud|travel=True)")
                print(belief)


