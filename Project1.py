#   Artificial Neural Network from scratch - Project 1
#   GEL521 - Machine Learning
#   Presented to: Dr. Hayssam SERHAN
#   Presented by:   Antonio HADDAD          - 202200238
#                   Elias-Charbel SALAMEH   - 202201047

import numpy as np
from numpy import random
import os
import pickle

class NeuralNetwork:
    def __init__(self, layers,learning_rate,momentum_factor,epochs): 
        self.layers             = layers
        self.num_layers         = len(layers)
        self.learning_rate      = learning_rate
        self.momentum_factor    = momentum_factor
        self.epochs             = epochs
        self.weights            = [np.random.randn(layers[i-1], layers[i]) for i in range(1,self.num_layers)]
        self.old_weights        = [np.zeros((layers[i-1], layers[i])) for i in range(1,self.num_layers)]

    def save_weights(self, filename="weights.npy"):
        np.save(filename, self.weights)

    def load_weights(self, filename="weights.npy"):
        if os.path.isfile(filename):
            self.weights = np.load(filename, allow_pickle=True)
        else:
            print("File not found. Loading default weights.")

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def forward_propagation(self, x):
        activations = [x]           # assign input to be activated
        zs = []                     # effective Y in the GEL521 course

        for w in self.weights:
            
            z = np.dot(activations[-1],w)
            zs.append(z)
            activations.append(self.sigmoid(z))

        return activations, zs      # V_layer,Y_layer

    def backward_propagation(self, x, y, activations, zs):
        deltas = [None] * self.num_layers
        gradients_w = [np.zeros(w.shape) for w in self.weights] 

        # Calculate output layer error
        deltas[-1] = (y - activations[-1]) * self.sigmoid_derivative(zs[-1])  # output_gradient

        # Backpropagation for hidden layers
        for l in range(self.num_layers - 2, 0, -1):  # Exclude input and output layers
            deltas[l] = np.dot(deltas[l+1], self.weights[l].T) * self.sigmoid_derivative(zs[l-1])

        # Compute gradients
        for l in range(self.num_layers - 1):
            gradients_w[l] = np.dot(activations[l].reshape(-1, 1), deltas[l+1].reshape(1, -1))

        return gradients_w



    def train(self, X, y):
        highest_accuracy = 0  # Variable to track the highest accuracy
        for epoch in range(self.epochs):
            total_gradients_w = [np.zeros(w.shape) for w in self.weights]
            correct_predictions = 0

            for x, y_target in zip(X, y):
                x = np.array(x).reshape(-1, 1).T
                y_target = np.array(y_target).reshape(-1, 1)

                activations, zs = self.forward_propagation(x)
                gradients_w = self.backward_propagation(x, y_target, activations, zs)

                total_gradients_w = [tw + gw for tw, gw in zip(total_gradients_w, gradients_w)]

                # Check if our prediction is correct
                prediction = 1 if activations[-1][0] >= 0.5 else 0
                if prediction == y_target:
                    correct_predictions += 1

            # Calculate accuracy
            accuracy = (correct_predictions / len(X))
            print(f"Epoch {epoch+1}/{self.epochs}, Accuracy: {accuracy*100:.2f}%")

            # Update highest_accuracy if current accuracy is higher
            if accuracy > highest_accuracy:
                highest_accuracy = accuracy

            # Update weights
            for l in range(self.num_layers - 1):
                self.weights[l] += self.momentum_factor * (self.weights[l] - self.old_weights[l]) + self.learning_rate * total_gradients_w[l]

            # Update old_weights
            self.old_weights = [w.copy() for w in self.weights]

        # Print highest accuracy reached
        print(f"Highest accuracy reached: {highest_accuracy * 100:.2f}%")
            
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from Project1 import NeuralNetwork

data = pd.read_csv("Project1/wisc_bc_data.csv")
X = data.iloc[:,2::].values     # Features, inputs
y = data.iloc[:,1].values       # Target labels

X=(X   -   X.min(axis=0)) /   (X.max(axis=0) - X.min(axis=0))   # normalize inputs

y = [1 if val == 'M' else 0 for val in y]           # transform 'B' and 'M' into 0 and 1

# Split data into training and testing sets (default 70% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42 )

# Further split training data into training and validation sets (optional)
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10)  # 10% validation

print("Training set size:", len(X_train))
print("Testing set size:", len(X_test))
#print("Validation set size (optional):", len(X_val) if X_val is not None else 



# Define network architecture
layers          = [30, 5, 4, 1]  
learning_rate   = 0.01
momentum_factor = 0.1
epochs          = 5000

# Create neural network
nn = NeuralNetwork(layers,learning_rate,momentum_factor,epochs)

# Train the network
nn.train(X_train, y_train)

# Evaluate the model on the train set
predictions = []
for x in X_train:
    x = np.array(x).reshape(-1, 1).T
    activations, _ = nn.forward_propagation(x)
    prediction = 1 if activations[-1][0] >= 0.5 else 0
    predictions.append(prediction)

# Calculate confusion matrix
true_positive = sum((p == 1) and (t == 1) for p, t in zip(predictions, y_train))
false_positive = sum((p == 1) and (t == 0) for p, t in zip(predictions, y_train))
false_negative = sum((p == 0) and (t == 1) for p, t in zip(predictions, y_train))
true_negative = sum((p == 0) and (t == 0) for p, t in zip(predictions, y_train))

# Display confusion matrix
confusion_matrix = np.array([['', 'PP', 'PN'],
                                ['AP', true_positive, false_negative],
                                ['AN', false_positive, true_negative]])

print("Confusion Matrix for the training set:")
for row in confusion_matrix:
    print("\t".join(map(str, row)))
    

predictions = []
for x in X:
    x = np.array(x).reshape(-1, 1).T
    activations, _ = nn.forward_propagation(x)
    prediction = 1 if activations[-1][0] >= 0.5 else 0
    predictions.append(prediction)

# Calculate confusion matrix
true_positive = sum((p == 1) and (t == 1) for p, t in zip(predictions, y))
false_positive = sum((p == 1) and (t == 0) for p, t in zip(predictions, y))
false_negative = sum((p == 0) and (t == 1) for p, t in zip(predictions, y))
true_negative = sum((p == 0) and (t == 0) for p, t in zip(predictions, y))

# Display confusion matrix
confusion_matrix = np.array([['', 'PP', 'PN'],
                                ['AP', true_positive, false_negative],
                                ['AN', false_positive, true_negative]])

print("Confusion Matrix for all the set:")
for row in confusion_matrix:
    print("\t".join(map(str, row)))