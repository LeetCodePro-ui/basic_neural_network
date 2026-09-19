#neural_network
import numpy as np
import sys

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

inputs = [1.2,5.1,2.1]

weights = np.array(
        [[3.1, 2.1, 8.7],
        [1.8,2.5,3],
        [2.7,2.7,-4]])

biases = [3,2,1]

goal = [0,1,0]

output = sigmoid(np.dot(inputs,weights) + biases)

print(output)
