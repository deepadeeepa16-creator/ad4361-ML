import random
import math

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# XOR dataset
X = [[0,0],
     [0,1],
     [1,0],
     [1,1]]

y = [[0],
     [1],
     [1],
     [0]]

# Initialize weights randomly
wh = [[random.random(), random.random()],
      [random.random(), random.random()]]

bh = [random.random(), random.random()]

wo = [[random.random()],
      [random.random()]]

bo = [random.random()]

learning_rate = 0.5
epochs = 5000

# Training
for _ in range(epochs):
    for i in range(len(X)):

        # Forward propagation
        hidden = []
        for j in range(2):
            h = X[i][0]*wh[0][j] + X[i][1]*wh[1][j] + bh[j]
            hidden.append(sigmoid(h))

        output = 0
        for j in range(2):
            output += hidden[j]*wo[j][0]
        output += bo[0]
        output = sigmoid(output)

        # Backpropagation
        error = y[i][0] - output
        d_output = error * sigmoid_derivative(output)

        d_hidden = []
        for j in range(2):
            d_hidden.append(d_output * wo[j][0] * sigmoid_derivative(hidden[j]))

        # Update output weights
        for j in range(2):
            wo[j][0] += hidden[j] * d_output * learning_rate
        bo[0] += d_output * learning_rate

        # Update hidden weights
        for j in range(2):
            wh[0][j] += X[i][0] * d_hidden[j] * learning_rate
            wh[1][j] += X[i][1] * d_hidden[j] * learning_rate
            bh[j] += d_hidden[j] * learning_rate

# Testing
print("Final Output:")
for i in range(len(X)):
    hidden = []
    for j in range(2):
        h = X[i][0]*wh[0][j] + X[i][1]*wh[1][j] + bh[j]
        hidden.append(sigmoid(h))

    output = 0
    for j in range(2):
        output += hidden[j]*wo[j][0]
    output += bo[0]
    output = sigmoid(output)

    print(X[i], "->", round(output, 3))
