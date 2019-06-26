from __future__ import print_function
import matplotlib.pyplot as plt
import pandas as pd  
import seaborn as sb
from ParticleSwarmOptimization import ParticleSwarmOptimizedNN
from utils import train_test_split, to_categorical, normalize, Plot
from sklearn.model_selection import train_test_split
from NeuralNetwork import NeuralNetwork
from layers import Activation, Dense
from loss_functions import CrossEntropy
from optimizers import Adam
import numpy
import pandas
# load dataset
'''
#print(df.head)
x = normalize(df)
X = pd.DataFrame(x).drop(labels="PPV", axis=1)
Y = pd.DataFrame(df.PPV)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
'''
#dataframe = pandas.read_csv("housing.csv", delim_whitespace=True, header=None)
#dataset = dataframe.values
# split into input (X) and output (Y) variables
#X = dataset[:,0:13]
#Y = dataset[:,13]
data = pd.read_csv("data.csv", sep = ',')
X = normalize(data)
#print(X)
df = pd.DataFrame(X)
X = df.drop(labels="PPV", axis = 1)
y = pd.DataFrame(df.PPV)
#print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

def model_builder(n_inputs, n_outputs):    

	model = NeuralNetwork(optimizer=Adam(), loss=CrossEntropy)
	model.add(Dense(12, input_shape=(n_inputs,)))
	model.add(Activation('relu'))
	model.add(Dense(8))
	model.add(Activation('relu'))
	model.add(Dense(1))
	model.add(Activation('relu'))

	return model

# Print the model summary of a individual in the population
print ("")
print(model_builder(n_inputs=X.shape[1], n_outputs=y.shape[1]).summary())


population_size = 100
n_generations = 10

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
X = pd.DataFrame(X).to_numpy()
y_test = pd.DataFrame(y_test).to_numpy()
X_train = pd.DataFrame(X_train).to_numpy()
y_train = pd.DataFrame(y_train).to_numpy()

#print(X)

model = ParticleSwarmOptimizedNN(population_size=population_size, 
                    inertia_weight=inertia_weight,
                    cognitive_weight=cognitive_weight,
                    social_weight=social_weight,
                    max_velocity=5,
                    model_builder=model_builder)

print("######################################################################")

#print(X_train, y_train)
model = model.evolve(X_train, y_train, n_generations=n_generations)

print("######################################################################")
_, loss, accuracy = model.test_on_batch(X_test, y_test)
#print(X_test, y_test)8iii
#print(loss)

inp_arr = pd.DataFrame([[0.624650,0.692476,0.000000,0.000000,0.218002,0.000000]])
inp_lab = pd.DataFrame([0.066579])
Ans, a1, a2 = model.test_on_batch(inp_arr, inp_lab)
print(Ans)
print ("Accuracy: %.1f%%" % float(100*accuracy))
