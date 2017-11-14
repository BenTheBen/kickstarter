#I got this code from https://machinelearnings.co/text-classification-using-neural-networks-f5cd7b8765c6
#I didn't change anything relevant to the structure of the network. It functions exactly as the author meant it to.
#I played around with the error threshold.
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import json
stemmer = LancasterStemmer()
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output
def clean_up_description(description):
    # tokenize the pattern
    description_words = nltk.word_tokenize(description)
    # stem each word
    description_words = [stemmer.stem(word.lower()) for word in description_words]
    return description_words
# return bag of words array: 0 or 1 for each word in the bag that exists in the description
def bow(description, words, show_details=False):
    # tokenize the pattern
    description_words = clean_up_description(description)
    # bag of words
    bag = [0]*len(words)  
    for s in description_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

def think(description, show_details=False):
    x = bow(description.lower(), words, show_details)
    if show_details:
        print ("description:", description, "\n bow:", x)
    # input layer is our bag of words
    l0 = x
    # matrix multiplication of input and hidden layer
    l1 = sigmoid(np.dot(l0, synapse_0))
    # output layer
    l2 = sigmoid(np.dot(l1, synapse_1))
    return l2
# probability threshold
ERROR_THRESHOLD = 0.2
# load our calculated synapse values
synapse_file = 'synapses.json' 
with open(synapse_file) as data_file: 
    synapse = json.load(data_file) 
    synapse_0 = np.asarray(synapse['synapse0']) 
    synapse_1 = np.asarray(synapse['synapse1'])
    words = np.asarray(synapse['words'])
    states = np.asarray(synapse['states'])

def classify(description, show_details=False):
    results = think(description, show_details)

    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD ] 
    results.sort(key=lambda x: x[1], reverse=True) 
    return_results =[[states[r[0]],r[1]] for r in results]
    print ("%s \n classification: %s" % (description, return_results))
    return return_results

#Lets see how it works!
classify("best experience")
classify("making hockey sticks best nerd")
classify("rawr")