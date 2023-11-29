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
parser = argparse.ArgumentParser(description='Experiment_basic')
parser.add_argument('--run', type=int, default=1, help='number of experiments to run')
parser.add_argument('--iteration', type=int, default=5000, help='number of iterations to simulate')
parser.add_argument('--exp', type=int, default=1, help='number of initial random exposure iterations')
parser.add_argument('--cycle_itr', type=int, default=50, help='number of iterations in one cycle')
parser.add_argument('--epoch', type=int, default=3, help='number of epochs to train')
parser.add_argument('--K', type=int, default=20, help='number of items to recommend')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--reg', type=float, default=1e-5, help='regularization')
parser.add_argument('--hidden', type=int, default=100, help='latent dimension')
parser.add_argument('--neg', type=int, default=5, help='negative sampling rate')
parser.add_argument('--data', type=str, default='ml1m', help='path to eval in the Data folder')

args = parser.parse_args()

# Load truth data and set experiment parameters
truth = np.load('./Data/' + args.data + '/truth.npy')
args.num_user = truth.shape[0]
args.num_item = truth.shape[1]
audience_size = np.sum(truth, axis=0)
item_sorted = np.argsort(audience_size)
truth_like = list(np.load('./Data/' + args.data + '/user_truth_like.npy', allow_pickle=True))

# Print total truth for reference
print('')
print('!' * 30 + ' Total truth ' + str(np.sum(truth)) + ' ' + '!' * 30)
print('')

# Calculate mainstream scores (MS_similarity) using your code
with open('./Data/' + args.data + '/info.pkl', 'rb') as f:
    info = pickle.load(f)
    num_user = info['num_user']
    num_item = info['num_item']

train_df = pd.read_csv('./Data/' + args.data + '/train_df.csv')

pos_user_array = train_df['userId'].values
pos_item_array = train_df['itemId'].values
train_mat = coo_matrix((np.ones(len(pos_user_array)), (pos_user_array, pos_item_array)), shape=(num_user, num_item)).toarray()
user_pop = np.sum(train_mat, axis=1)

Jaccard_mat = np.matmul(train_mat, train_mat.T)
deno = user_pop.reshape((-1, 1)) + user_pop.reshape((1, -1)) - Jaccard_mat + 1e-7
Jaccard_mat /= deno
Jaccard_mat = Jaccard_mat + np.eye(num_user) * -9999
Jaccard_mat = Jaccard_mat[np.where(Jaccard_mat > -1)].reshape((num_user, num_user - 1))
MS_similarity = np.mean(Jaccard_mat, axis=1)
with open('./Data/' + args.data + '/MS_similarity.npy', "wb") as f:
    np.save(f, MS_similarity)

# Initialize lists to store results
itr_cumulated_click_count_list = []
itr_GC_TPR_list = []

# Initialize a list to store MS_similarity values for each run
MS_similarity_list = []

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
    itr_click_item = simulation.run_simulation()  # Run the main simulation

    itr_cumulated_click_count = []
    itr_item_click = np.zeros((args.iteration, args.num_item))

    # Process simulation results 
    for itr in range(args.iteration):
        click_item = itr_click_item[itr]
        itr_item_click[itr, click_item] = 1.
        itr_cumulated_click_count.append(
            len(click_item) if itr == 0 else len(click_item) + itr_cumulated_click_count[-1])

    for itr in range(1, args.iteration):
        itr_item_click[itr, :] += itr_item_click[itr - 1, :]

    itr_item_click /= (audience_size - init_popularity).reshape((1, -1))

    itr_GC_TPR = []
    
    # Calculate GC-TPR (Groupwise Click-Through Rate) at each iteration
    for itr in range(args.iteration):
        a = itr_item_click[itr, item_sorted]
        gc = np.sum(((np.arange(len(a)) + 1.) * 2 - len(a) - 1) * a) / (len(a) * np.sum(a))
        itr_GC_TPR.append(gc)

    # Store results for analysis
    itr_cumulated_click_count_list.append(itr_cumulated_click_count)
    itr_GC_TPR_list.append(itr_GC_TPR)

# Calculate mainstream bias (MS_similarity) statistics across runs
MS_similarity_mean = np.mean(MS_similarity_list, axis=0)
MS_similarity_std = np.std(MS_similarity_list, axis=0)

# Calculate the mean and standard deviation of GC-TPR across runs
itr_cumulated_click_count_mean = np.mean(itr_cumulated_click_count_list, axis=0)
itr_cumulated_click_count_std = np.std(itr_cumulated_click_count_list, axis=0)
itr_GC_TPR_mean = np.mean(itr_GC_TPR_list, axis=0)
itr_GC_TPR_std = np.std(itr_GC_TPR_list, axis=0)

# Save or visualize the mainstream bias (MS_similarity) statistics

# Plot the mainstream bias (MS_similarity) statistics
plt.figure(figsize=(12, 6))

# # Plot MS_similarity
# plt.subplot(1, 2, 1)
# plt.plot(range(len(MS_similarity_mean)), MS_similarity_mean, marker='.', linewidth=1.5, markersize=1)
# plt.fill_between(range(len(MS_similarity_mean)), MS_similarity_mean - MS_similarity_std,
#                  MS_similarity_mean + MS_similarity_std, alpha=0.5)
# plt.ylabel('MS_similarity')
# plt.xlabel('Iteration')
# plt.title('Mainstream Similarity Over Time')
# plt.grid(True)

# Create an array of iteration numbers for the x-axis
iterations = np.arange(args.iteration)

# Ensure that MS_similarity_mean has the same shape as iterations
MS_similarity_mean = np.repeat(MS_similarity_mean, len(iterations))

# Plot the mainstream bias (MS_similarity) statistics
plt.figure(figsize=(12, 6))

# Plot MS_similarity
plt.subplot(1, 2, 1)
plt.plot(iterations, MS_similarity_mean, marker='.', linewidth=1.5, markersize=1)
plt.fill_between(iterations, MS_similarity_mean - MS_similarity_std,
                 MS_similarity_mean + MS_similarity_std, alpha=0.5)
plt.ylabel('MS_similarity')
plt.xlabel('Iteration')
plt.title('Mainstream Similarity Over Time')
plt.grid(True)


# Plot GC-TPR
plt.subplot(1, 2, 2)
plt.plot(range(len(itr_GC_TPR_mean)), itr_GC_TPR_mean, marker='.', linewidth=1.5, markersize=1)
plt.fill_between(range(len(itr_GC_TPR_mean)), itr_GC_TPR_mean - itr_GC_TPR_std,
                 itr_GC_TPR_mean + itr_GC_TPR_std, alpha=0.5)
plt.ylabel('GC_TPR')
plt.xlabel('Iteration')
plt.title('GC_TPR Over Time')
plt.grid(True)

plt.tight_layout()
plt.show()

# Optionally, save the mainstream bias (MS_similarity) statistics to a file
# with open('./Data/' + args.data + '/MS_similarity_mean.npy', "wb") as f:
#     np.save(f, MS_similarity_mean)
# with open('./Data/' + args.data + '/MS_similarity_std.npy', "wb") as f:
#     np.save(f, MS_similarity_std)

