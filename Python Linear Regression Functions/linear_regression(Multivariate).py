# Import necessary libraries
from random import SystemRandom
from typing import Iterable
import numpy as np
import matplotlib.pyplot as plt
from noplot_linear_regression import Linear_Regression
import pandas as pd
import os

# Type alias for representing n-space vector data-points
VectorN = Iterable[float]

# Instantiate the random number generator class
rand = SystemRandom()

# Constants that store the dimensionality of the data-space
DIMENSIONALITY = 7  # You can change this to any positive integer

# Generate random coefficients for the linear equation
COEFFICIENTS = [rand.random() for _ in range(DIMENSIONALITY)]
INTERCEPT = rand.uniform(0, 100)

# Generate the original line and observations with uncorrelated predictor variables
original_line = [
    tuple(rand.random() for _ in range(DIMENSIONALITY)) + (sum(COEFFICIENTS[i] * rand.random() for i in range(DIMENSIONALITY)) + INTERCEPT,)
    for _ in range(100)
]

observations = [
    tuple(rand.random() for _ in range(DIMENSIONALITY)) + (sum(COEFFICIENTS[i] * rand.random() for i in range(DIMENSIONALITY)) + INTERCEPT + rand.gauss(0, 10),)
    for _ in range(100)
]

# Prepare the data for the linear regression model
X = np.array([list(x[:-1]) for x in observations])
y = np.array([x[-1] for x in observations])

def Multivariable_Linear_Regression(X, y):
    """
    Performs multivariable linear regression on the given dataset.
    """
    # Calculate the means of X and y
    X_mean = np.mean(X, axis=0)
    y_mean = np.mean(y)

    # Center the data
    X_centered = X - X_mean
    y_centered = y - y_mean

    # Calculate beta
    beta = np.linalg.inv(X_centered.T @ X_centered) @ X_centered.T @ y_centered

    # Calculate alpha
    alpha = y_mean - np.dot(beta, X_mean)

    return alpha, beta

# Perform multivariable linear regression
alpha_estimate, beta_estimate = Multivariable_Linear_Regression(X, y)

# Print the estimated coefficients and intercept
print("Estimated coefficients:", beta_estimate)
print("Estimated intercept:", alpha_estimate)

# Calculate the predicted values
y_pred = alpha_estimate + np.dot(X, beta_estimate)

# Calculate the residuals
residuals = y - y_pred

# Create a directory to save the plots
if not os.path.exists('plots'):
    os.makedirs('plots')

# Create a residual plot
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='-')
plt.title('Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.savefig('plots/Residual_Plot.png')
plt.show()
plt.close()

# Create a DataFrame for pairwise scatter plot
df = pd.DataFrame(X, columns=[f'Variable {i+1}' for i in range(DIMENSIONALITY)])
df['Target'] = y


# Create a pairwise scatter plot and save it to the directory
for i in range(DIMENSIONALITY):
    for j in range(DIMENSIONALITY):
        plt.figure(figsize=(10, 6))
        plt.scatter(df.iloc[:, i], df.iloc[:, j])
        plt.xlabel(f'Variable {i+1}')
        plt.ylabel(f'Variable {j+1}')
        plt.savefig(f'plots/Variable_{i+1}_vs_Variable_{j+1}.png')
        plt.close()

def plot_selected_pair(i, j):
    """
    Plots the selected pair of variables.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(df.iloc[:, i-1], df.iloc[:, j-1])
    plt.xlabel(f'Variable {i}')
    plt.ylabel(f'Variable {j}')
    plt.savefig(f'plots/Selected_Pair_{i}_{j}.png')
    plt.show()
    plt.close()

def plot_selected_pair_with_fit(i, j):
    """
    Plots the selected pair of variables with a best fit line.
    """
    plt.figure(figsize=(10, 6))
    x = df.iloc[:, i-1]
    y = df.iloc[:, j-1]
    plt.scatter(x, y)
    slope, intercept = Linear_Regression(list(zip(x, y)))
    plt.plot(x, slope * x + intercept, color='red')  # Add the best fit line
    plt.xlabel(f'Variable {i}')
    plt.ylabel(f'Variable {j}')
    plt.savefig(f'plots/Selected_Pair_with_Fit_{i}_{j}.png')
    plt.show()
    plt.close()

# Call these functions to see the pairwise plots (with or without best fit line depending on your data)
#plot_selected_pair(1,2)
#plot_selected_pair_with_fit(3,2)
