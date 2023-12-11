# Import dependencies
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
import warnings; warnings.simplefilter('ignore')  # Ignore warnings for cleaner output
# import time
import numpy as np
import argparse
# import utility
from Simulation_basic import Simulation
import pickle
import pandas as pd
# from tqdm import tqdm
from math import log
from scipy.sparse import coo_matrix
# import seaborn as sns
import matplotlib.pyplot as plt
# from scipy.stats import skew
# from scipy.stats import mode
# from sklearn.neighbors import LocalOutlierFactor
# Parse command-line arguments
# args = parser.parse_args()

# Define the default values for arguments
default_args = {
    'run': 1,
    'iteration': 1000,
    'exp': 1,
    'cycle_itr': 50,
    'epoch': 20,
    'K': 20,
    'lr': 0.001,
    'reg': 1e-5,
    'hidden': 100,
    'neg': 5,
    'data': 'ml1m'
}

# Create the args object with default values
args = argparse.Namespace(**default_args)
# Parse command-line arguments
# args = parser.parse_args()

# Define the default values for arguments
default_args = {
    'run': 1,
    'iteration': 1000,
    'exp': 1,
    'cycle_itr': 50,
    'epoch': 20,
    'K': 20,
    'lr': 0.001,
    'reg': 1e-5,
    'hidden': 100,
    'neg': 5,
    'data': 'ml1m'
}

# Create the args object with default values
args = argparse.Namespace(**default_args)

# # Calculate mainstream scores (MS_similarity) using your code
# train_df = pd.read_csv('./Data/' + args.data + '/train_df.csv')

# # Calculate user popularity
# pos_user_array = train_df['userId'].values
# pos_item_array = train_df['itemId'].values
# train_mat = coo_matrix((np.ones(len(pos_user_array)), (pos_user_array, pos_item_array)), shape=(num_user, num_item)).toarray()
# user_pop = np.sum(train_mat, axis=1)

# # Calculate standard deviation of user interactions
# user_stddev = np.std(train_mat, axis=1)

# # Save the user standard deviations to a file (adjust the path accordingly)
# with open(f'./Data/{args.data}/user_stddev.npy', "wb") as f:
#     np.save(f, user_stddev)

# # Calculate Jaccard similarity matrix
# Jaccard_mat = np.matmul(train_mat, train_mat.T)
# deno = user_pop.reshape((-1, 1)) + user_pop.reshape((1, -1)) - Jaccard_mat + 1e-7
# Jaccard_mat /= deno
# Jaccard_mat = Jaccard_mat + np.eye(num_user) * -9999
# Jaccard_mat = Jaccard_mat[np.where(Jaccard_mat > -1)].reshape((num_user, num_user - 1))

# # Calculate Mainstream Similarity (MS_similarity) by taking the mean along axis 1
# MS_similarity = np.mean(Jaccard_mat, axis=1)

# # Save the MS similarity to a file (adjust the path accordingly)
# with open(f'./Data/{args.data}/MS_similarity.npy', "wb") as f:
#     np.save(f, MS_similarity)


# Since we're not using train_df, let's create a user-item matrix from truth
user_item_matrix = truth.copy()  # Assuming truth already represents user-item interactions

# Calculate user popularity based on the user-item matrix
user_pop = np.sum(user_item_matrix, axis=1)

# Calculate Jaccard similarity matrix based on the user-item matrix
Jaccard_mat = np.matmul(user_item_matrix, user_item_matrix.T)
deno = user_pop.reshape((-1, 1)) + user_pop.reshape((1, -1)) - Jaccard_mat + 1e-7
Jaccard_mat /= deno
np.fill_diagonal(Jaccard_mat, 0)  # Set diagonal to zero to exclude self-similarity

# Calculate Mainstream Similarity (MS_similarity) by taking the mean along axis 1
MS_similarity = np.mean(Jaccard_mat, axis=1)

# Save the MS similarity to a file (adjust the path accordingly)
with open(f'./Data/{args.data}/MS_similarity.npy', "wb") as f:
    np.save(f, MS_similarity)

print("Mainstream scores: " + str(len(MS_similarity)) + "\n")
print("Args Number of Users: " + str(args.num_user) + "\n")
# Initialize a list to store Gini coefficients after each epoch
gini_coefficients = []

# Run the experiment for a specified number of runs
for r in range(args.run):
    print('')
    print('#' * 100)
    print('#' * 100)
    print(' ' * 50 + ' Experiment run ' + str(r + 1) + ' ' * 50)
    print('#' * 100)
    print('#' * 100)

    # Initialize the simulation with provided arguments and data
    simulation = Simulation(args, truth, truth_like)
    init_popularity = simulation.initial_iterations()  # Perform initial iterations to gather feedback
    gini_coefficients.append(simulation.run_simulation())  # Run the main simulation
    # Calculate the average of each corresponding element across lists
print(gini_coefficients)
average_gini_coefficients = np.mean(np.array(gini_coefficients), axis=0)

# Visualize the averaged Gini coefficients
plt.figure(figsize=(10, 6))
x_values = range(len(average_gini_coefficients))
plt.plot(x_values, average_gini_coefficients, marker='o', linestyle='-', color='b')
plt.title('Average Gini Coefficients Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Average Gini Coefficient')
plt.grid(True)
print(average_gini_coefficients)
plt.show()