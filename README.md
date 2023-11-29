# Mainstream-Bias-in-Dynamic-Recommendation
Code for the paper -- Mainstream Bias in Dynamic Recommendation

## Data
We put the pre-processed MovieLens 1M dataset in the 'Data' folder, where 1000 users are randomly selected for the simulation experiments. We run an MF model with cross entropy loss to complete the user-item relevance matrix to get the ground truth data for the simulation experiments.

## Requirements
python 3, tensorflow 1.14.0, numpy, pandas

## Excution
To run the similation experiment with different settings, use the 'Mainstream_Analysis.ipynb' file. Experiment settings can be changed from an input box with the simulation arguments.

**Experiment_basic**: with vanilla MF, position bias, closed feedback loop, better negative sampling.  


After running the simulations, to plot the results, go to the 'Experiment' folders in './Data/ml1m', and run the 'analysis.ipynb' to see the ploting results. For example, to see the results of Experiment_basic, go to './Data/ml1m/Experiment_basic', and run 'Mainstream_Analysis.ipynb'.
