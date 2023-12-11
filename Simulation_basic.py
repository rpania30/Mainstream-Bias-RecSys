import tensorflow as tf
import time
import numpy as np
import pickle
import argparse
import pandas as pd
import utility
from operator import itemgetter
from tqdm import tqdm
from MF import MF
from scipy.sparse import coo_matrix

# import warnings; warnings.simplefilter('ignore') 
# Ignores Warnings for nicer Plots. Disable for Debugging

class Simulation:
    def __init__(self, args, truth, truth_like, record=False):
        # Initialize the simulation with the provided arguments
        print(args)

        self.record = record
        self.data = args.data

        self.truth = truth
        self.truth_like = truth_like

        self.num_user = args.num_user
        self.num_item = args.num_item

        self.exp = args.exp
        self.cycle_itr = args.cycle_itr
        self.iteration = args.iteration
        self.K = args.K
        self.epoch = args.epoch
        self.lr = args.lr
        self.reg = args.reg
        self.hidden = args.hidden
        self.neg = args.neg

        self.feedback = [] # Store user feedback (clicks)
        self.neg_feedback = [] # Store negative feedback
        self.user_feedback = [[] for _ in range(self.num_user)] # Store feedback for each user

        self.audience_size = np.sum(truth, axis=0) # Calculate the size of the audience for each item
        self.item_sorted = np.argsort(self.audience_size) # Sort items by audience size

        self.truth_unclick = truth.copy() # Initialize truth_unclick matrix

        self.feedback_expose_prob = [] # Store exposure probabilities for feedback items

    def initial_iterations(self):
        # Perform initial random iterations
        print('*' * 30 + ' Start initial random iterations ' + '*' * 30)

        init_click_mat = np.zeros((self.num_user, self.num_item)) # Initialize a matrix to track clicks

        expose_count = np.ones(self.num_item) # Initialize exposure count for items
        feedback_item_list = [] # Store feedback item list
        for e in range(self.exp):
            print('-' * 10 + ' Iteration ' + str(e + 1) + ' ' + '-' * 10)
            for u in tqdm(range(self.num_user)):
                u_feedback = self.user_feedback[u] # Get user's existing feedback
                u_truth_like = self.truth_like[u] # Get user's truth-like preferences

                scores = np.random.random(self.num_item) # Generate random scores for items
                scores[u_feedback] = -9999. # Set scores for items that are already clicked to a very low value

                # Select the top K items based on scores
                topK_iid = np.argpartition(scores, -self.K)[-self.K:]
                topK_iid = topK_iid[np.argsort(scores[topK_iid])[-1::-1]]

                randomK = np.random.random(self.K)
                for k in range(self.K):
                    iid = topK_iid[k]
                    expose_count[iid] += 1.
                    if iid in u_truth_like and randomK[k] <= utility.position_bias[k]:
                        self.user_feedback[u].append(iid)
                        self.feedback.append([u, iid, k])
                        init_click_mat[u, iid] = 1.
                        self.truth_unclick[u, iid] = 0
                        feedback_item_list.append(iid)
                    else:
                        self.neg_feedback.append([u, iid, k])

        expose_prob = expose_count / (self.exp * self.num_user + 1)
        self.feedback_expose_prob += list(expose_prob[feedback_item_list])

        print('!' * 100)
        print('Generate ' + str(len(self.feedback)) + ' records.')
        print('!' * 100)

        if self.record:
            np.save('./Data/' + self.data + '/Experiment_basic/init_click_mat.npy', init_click_mat)

        self.init_popularity = np.sum(init_click_mat, axis=0)
        return self.init_popularity

    def run_simulation(self):
        print('*' * 30 + ' Train MF until converge ' + '*' * 30)
        bs = int(len(self.feedback) * (self.neg + 1) / 50.)
        print('Update bs to ' + str(bs))
        mf = MF(self.lr, self.reg, self.hidden, self.neg, self.num_user, self.num_item)
        feedback_array = np.array(self.feedback)
        neg_feedback_array = np.array(self.neg_feedback)
        for j in range(self.epoch):
            mf.train(feedback_array, neg_feedback_array, j, bs, verbose=True)
            mf.record()

        print('*' * 30 + ' Start simulation ' + '*' * 30)

        # Load the Mainstream scores
        MS_similarity_file_path = f'./Data/{self.data}/MS_similarity.npy'
        MS_similarity = np.load(MS_similarity_file_path)
        sorted_indices = np.argsort(MS_similarity)[::-1]
        
        # Initialize lists to store results for each epoch
        gini_coefficients = []
        itr_user_click_item = []
        itr_cumulated_click_count_list = []
        itr_GC_TPR_list = []

        for epoch in range(self.epoch):
            print('*' * 30 + f' Epoch {epoch} ' + '*' * 30)

            itr_rec_item = []
            itr_user = []  # Store user IDs for this epoch
            user_list = np.random.randint(self.num_user, size=self.num_user) # UIDs
            user_count = np.zeros(self.num_user)
            item_click = np.zeros(self.num_item)
            last_time = time.time()
            itr = 0

            for _ in range(int(self.iteration / self.cycle_itr)):
                for _ in tqdm(range(self.cycle_itr)):
                    uid = user_list[itr]
                    u_feedback = self.user_feedback[uid]
                    u_truth_like = self.truth_like[uid]

                    user_count[uid] += 1
                    scores = mf.predict_scores(uid)
                    scores[u_feedback] = -9999.
                    topK_iid = np.argpartition(scores, -self.K)[-self.K:]
                    topK_iid = topK_iid[np.argsort(scores[topK_iid])[-1::-1]]
                    itr_user.append(uid)
                    itr_rec_item.append(topK_iid)

                    randomK = np.random.random(self.K)
                    click_item = []

                    for k in range(self.K):
                        iid = topK_iid[k]
                        if iid in u_truth_like and randomK[k] <= utility.position_bias[k]:
                            self.user_feedback[uid].append(iid)
                            self.feedback.append([uid, iid, k])
                            click_item.append(iid)
                            item_click[iid] += 1
                            self.truth_unclick[uid, iid] = 0
                        else:
                            self.neg_feedback.append([uid, iid, k])

                    itr_user_click_item.append(click_item)
                    itr_user.append(uid)
                    itr += 1
                    # Remove clicked items from: self.truth_like
                    self.truth_like[uid] = [item for item in self.truth_like[uid] if item not in click_item]

            # Calculate TPR (True Positive Rate) for the current epoch
            user_TPR = np.zeros(self.num_user)
            user_Precision = np.zeros(self.num_user)
            user_DCG = np.zeros(self.num_user)
            user_NDCG = np.zeros(self.num_user)
            for user in range(self.num_user):
                clicked_items = itr_user_click_item[user]
                total_positives = len(self.truth_like[user])
                if total_positives > 0:
                    true_positives = len(clicked_items)
                    #Recall, Precision, DCG, NDCG
                    user_TPR[user] = true_positives / total_positives
                    user_Precision[user] = true_positives / len(clicked_items) if len(clicked_items) > 0 else 0
                    user_DCG[user] = sum(1 / np.log2(i + 2) for i, item in enumerate(clicked_items))
                    ideal_dcg = sum(1 / np.log2(i + 2) for i in range(total_positives))
                    user_NDCG[user] = user_DCG[user] / ideal_dcg if ideal_dcg > 0 else 0

            # Calculate Gini coefficient for this epoch
            print(user_TPR.dtype, sorted_indices.dtype)
            a = user_TPR[sorted_indices]
            gc = np.sum(((np.arange(len(a)) + 1.) * 2 - len(a) - 1) * a) / (len(a) * np.sum(a))
            gini_coefficients.append(gc)
            
            # new analysis code
            # Cumulated click count and item click calculation for the epoch
            itr_cumulated_click_count = []
            itr_item_click = np.zeros((self.num_user, self.num_item))
            for uid in itr_user:
                click_item = self.user_feedback[uid]
                itr_item_click[uid, click_item] = 1.
                cum_click_count = len(click_item) if uid == 0 else len(click_item) + itr_cumulated_click_count[-1]
                itr_cumulated_click_count.append(cum_click_count)
            for uid in range(1, self.num_user):
                itr_item_click[uid, :] += itr_item_click[uid - 1, :]
            itr_item_click /= (self.audience_size - self.init_popularity).reshape((1, -1))

            # GC_TPR calculation for the epoch
            itr_GC_TPR = []
            for uid in range(self.num_user):
                a = itr_item_click[uid, self.item_sorted]
                gc_tpr = np.sum(((np.arange(len(a)) + 1.) * 2 - len(a) - 1) * a) / (len(a) * np.sum(a))
                itr_GC_TPR.append(gc_tpr)

            # Store the results for the epoch
            itr_cumulated_click_count_list.append(itr_cumulated_click_count)
            itr_GC_TPR_list.append(itr_GC_TPR)

            print('#' * 10
                  + ' The iteration %d, up to now total %d clicks, GC=%.4f, this cycle used %.2f s) '
                  % (itr, len(self.feedback), gc, time.time() - last_time)
                  + '#' * 10)

            last_time = time.time()

            bs = int(len(self.feedback) * (self.neg + 1) / 50.)
            print('Update bs to ' + str(bs))
            mf.reset()
            feedback_array = np.array(self.feedback)
            neg_feedback_array = np.array(self.neg_feedback)
            for j in tqdm(range(self.epoch)):
                mf.train(feedback_array, neg_feedback_array, j, bs, verbose=False)
            mf.record()
            print('')

        if self.record:
            np.save('./Data/' + self.data + '/Experiment_basic/itr_user.npy', np.array(itr_user))
            np.save('./Data/' + self.data + '/Experiment_basic/itr_rec_item.npy', np.array(itr_rec_item))
            np.save('./Data/' + self.data + '/Experiment_basic/itr_click_item.npy', np.array(itr_user_click_item, dtype=object))

        return gini_coefficients