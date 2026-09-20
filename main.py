#neural_network
import numpy as np
import sys

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

inputs = [1.9,5,3.1]

weights = np.random.uniform(-0.5,0.5, size=(3,3))

biases = np.zeros(3)

goal = [0,1,0]
learning_rate = 0.1


for i in range(100000):
    hidden_layer0 = sigmoid(np.dot(inputs,weights) + biases)
    distance = hidden_layer0 - goal
    #delta = wie sehr liegt das neuron falsch?
    delta = distance * hidden_layer0 * (1 - hidden_layer0)
    weights = weights - learning_rate * np.outer(inputs,delta)
    biases = biases -learning_rate * delta


print(f"Expected outcome : {goal}")
print(f"Actuall outcome : {hidden_layer0}")
