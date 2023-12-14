# Mainstream-Bias-in-Dynamic-Recommendation
Paper for the Code -- Mainstream Bias in Dynamic Recommendation
https://docs.google.com/document/d/1JEPyXzrLPNtUJIUYfAHk_0Ym9-fj5Xx2kMiHor0GE0c/edit

## Data
We put the pre-processed MovieLens 1M dataset in the 'Data' folder, where 1000 users are randomly selected for the simulation experiments. We run an MF model with cross entropy loss to complete the user-item relevance matrix to get the ground truth data for the simulation experiments.

## Requirements
python 3, tensorflow 1.14.0, numpy, pandas

## Excution
To run the similation experiment with different settings, use the 'Mainstream_Experiment_Basic.ipynb' file. Experiment settings can be changed from an input box with the simulation arguments.

**Simulation_basic**: with vanilla MF, position bias, closed feedback loop, better negative sampling.  


To see the results of of the experiment, go to './Data/ml1m', and look at the cooresponding data files.
