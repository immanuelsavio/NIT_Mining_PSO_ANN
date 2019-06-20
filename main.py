from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  
import seaborn as sb
from ParticleSwarmOptimization import ParticleSwarmOptimizedNN
from utils import train_test_split, to_categorical, normalize, Plot
from sklearn.model_selection import train_test_split, datasets
from NeuralNetwork import NeuralNetwork
from layers import Activation, Dense
from loss_functions import CrossEntropy
from optimizers import Adam

def main():

    df = pd.read_csv("data.csv",sep=',')
    #print(df.head)
    x = normalize(df)
    X = pd.DataFrame(x).drop(labels="PPV", axis=1)
    y = pd.DataFrame(df.PPV)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
    #print(X_train.head)
    #print(y_train.head)
    
    data = datasets.load_iris()
    Y = data.target
    print(Y.head)
    Y = to_categorical(Y.astype("int"))
    print(Y)
    #print(y.shape[1])
'''
   # Model builder
    def model_builder(n_inputs, n_outputs):    
        model = NeuralNetwork(optimizer=Adam(), loss=CrossEntropy)
        model.add(Dense(6, input_shape=(n_inputs,)))
        model.add(Activation('relu'))
        model.add(Dense(12))
        model.add(Activation('relu'))
        model.add(Dense(1))

        return model

    # Print the model summary of a individual in the population
    print ("")
    model_builder(n_inputs=X.shape[1], n_outputs=y.shape[1]).summary()

    population_size = 100
    n_generations = 10

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

    inertia_weight = 0.8
    cognitive_weight = 0.8
    social_weight = 0.8

    print ("Population Size: %d" % population_size)
    print ("Generations: %d" % n_generations)
    print ("")
    print ("Inertia Weight: %.2f" % inertia_weight)
    print ("Cognitive Weight: %.2f" % cognitive_weight)
    print ("Social Weight: %.2f" % social_weight)
    print ("")

    model = ParticleSwarmOptimizedNN(population_size=population_size, 
                        inertia_weight=inertia_weight,
                        cognitive_weight=cognitive_weight,
                        social_weight=social_weight,
                        max_velocity=5,
                        model_builder=model_builder)

    print("Model Built---------------------")

    model = model.evolve(X_train, y_train, n_generations=n_generations)

    print("----------------------Model Evolved")

    loss, accuracy = model.test_on_batch(X_test, y_test)

    print ("Accuracy: %.1f%%" % float(100*accuracy))

    # Reduce dimension to 2D using PCA and plot the results
    y_pred = np.argmax(model.predict(X_test), axis=1)
    #Plot().plot_in_2d(X_test, y_pred, title="Particle Swarm Optimized Neural Network", accuracy=accuracy, legend_labels=range(y.shape[1]))
'''
if __name__ == "__main__":
    main()