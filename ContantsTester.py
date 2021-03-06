#we will use this to figure out what the best constant are for the networks
#I got this code from https://machinelearnings.co/text-classification-using-neural-networks-f5cd7b8765c6
#I deleted the part of the network that output the dates and times with the synapses, and I also played around with alpha, the number of hidden neurons, and the drop percentage.
#alpha and dropout between 0 and 1
#hidden neurons between 1 and # of words in bag of words
#epochs is number of training loops
#dropout = true
import nltk
from nltk.stem.lancaster import LancasterStemmer
import os
import numpy as np
import time
import json
import datetime
from nltk.corpus import stopwords
stemmer = LancasterStemmer()
# Here is where we load our training data
with open('final_project.json') as json_data:
	training_data = json.load(json_data)[:3]
words = []
states = []
documents = []
# loop through each description in our training data
for pattern in training_data:
    # add to our words list
    words.extend(pattern[0])
    # add to documents in our corpus
    documents.append((pattern[0], pattern[1]))
    # add to our states list
    if pattern[1] not in states:
        states.append(pattern[1])
#remove duplicates
words = list(set(words))
# remove duplicates (should be unneccesary)
states = list(set(states))

# create our training data
training = []
output = []
# create an empty array for our output
output_empty = [0] * len(states)

# training set, bag of words for each description
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # create our bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    training.append(bag)
    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[states.index(doc[1])] = 1
    output.append(output_row)

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)
 
def train(X, y, hidden_neurons=10, alpha=1, epochs=10000, dropout=True, dropout_percent=0.5):
    np.random.seed(1)
    #print ("Input matrix: %sx%s    Output matrix: %sx%s" % (len(X),len(X[0]),1, len(states)) )
    last_mean_error = 1
    # randomly initialize our weights with mean 0
    synapse_0 = 2*np.random.random((len(X[0]), hidden_neurons)) - 1
    synapse_1 = 2*np.random.random((hidden_neurons, len(states))) - 1

    prev_synapse_0_weight_update = np.zeros_like(synapse_0)
    prev_synapse_1_weight_update = np.zeros_like(synapse_1)

    synapse_0_direction_count = np.zeros_like(synapse_0)
    synapse_1_direction_count = np.zeros_like(synapse_1)
        
    for j in iter(range(epochs+1)):

        # Feed forward through layers 0, 1, and 2
        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0, synapse_0))
                
        if(dropout):
            layer_1 *= np.random.binomial([np.ones((len(X),hidden_neurons))],1-dropout_percent)[0] * (1.0/(1-dropout_percent))

        layer_2 = sigmoid(np.dot(layer_1, synapse_1))

        # how much did we miss the target value?
        layer_2_error = y - layer_2

        if (j% 10000) == 0 and j > 5000:
            # if this 10k iteration's error is greater than the last iteration, break out
            if np.mean(np.abs(layer_2_error)) < last_mean_error:
                #print ("delta after "+str(j)+" iterations:" + str(np.mean(np.abs(layer_2_error))) )
                last_mean_error = np.mean(np.abs(layer_2_error))
            else:
                #print ("break:", np.mean(np.abs(layer_2_error)), ">", last_mean_error )
                break
                
        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)

        # how much did each l1 value contribute to the l2 error (according to the weights)?
        layer_1_error = layer_2_delta.dot(synapse_1.T)

        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
        
        synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
        synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))
        
        if(j > 0):
            synapse_0_direction_count += np.abs(((synapse_0_weight_update > 0)+0) - ((prev_synapse_0_weight_update > 0) + 0))
            synapse_1_direction_count += np.abs(((synapse_1_weight_update > 0)+0) - ((prev_synapse_1_weight_update > 0) + 0))        
        
        synapse_1 += alpha * synapse_1_weight_update
        synapse_0 += alpha * synapse_0_weight_update
        
        prev_synapse_0_weight_update = synapse_0_weight_update
        prev_synapse_1_weight_update = synapse_1_weight_update
    return(np.mean(np.abs(layer_2_error)))

X = np.array(training)
y = np.array(output)
neuronsMultiplier = 10
dropoutRange = 20
alphaRange = 20
bestError = 1
bestAlpha = 0.2
bestNeurons = 0
bestDropout = 0.5
for neurons in range(int(len(words)/neuronsMultiplier)):
	for alphaA in range(alphaRange):
		for dropoutP in range(dropoutRange):
			if neurons!=0 and alphaA!=0 and dropoutP!=0:
				totalError=train(X, y, hidden_neurons=neurons*neuronsMultiplier, alpha=(alphaA/alphaRange), dropout_percent=(dropoutP/dropoutRange)) #dont mess with epochs. also, lets use dropout=true
				print("Neurons: %r --- Alpha: %r --- Dropout: %r --- Error: %r" %((neurons*neuronsMultiplier),(alphaA/alphaRange),(dropoutP/dropoutRange),totalError))
				if(totalError<bestError):
					bestAlpha = (alphaA/alphaRange)
					bestNeurons = neurons*neuronsMultiplier
					bestDropout = (dropoutP/dropoutRange)
					print(bestAlpha, bestNeurons, bestDropout)
					bestError = totalError
with open("BestConstants.txt", 'w') as outfile:
	outfile.write("Best Constants... \n Alpha: %r \n Dropout Percent: %r \n Hidden Neurons: %r" %(bestAlpha,bestDropout,bestNeurons))
print ("saved constants to BestConstants.txt")