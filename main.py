from __future__ import print_function
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

from ParticleSwarmOptimization import ParticleSwarmOptimizedNN
from utils import train_test_split, to_categorical, normalize, Plot
from NeuralNetwork import NeuralNetwork
#from mlfromscratch.deep_learning.layers import Activation, Dense
#from mlfromscratch.deep_learning.loss_functions import CrossEntropy
#from mlfromscratch.deep_learning.optimizers import Adam
