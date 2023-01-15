import numpy as np
import os


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def fpass(inputs, weights, biases):
    return np.dot(inputs, weights) + biases

def d_sigmoid(x): # derivative of sigmoid
    return (np.exp(-x))/((np.exp(-x)+1)**2)

def save_results(l_hiddenLayers, batch_size, lrate, epoch, time_taken_in_s):
    os.chdir(os.path.join(os.getcwd(), 'Training Results'))
    with open('Training Results.txt', 'a') as file:
        line = f'Hidden Layers: {l_hiddenLayers} | Batch Size: {batch_size} | Learning Rate: {lrate} | Epoch: {epoch} | Time Taken (in s): {time_taken_in_s} \n \n'
        file.write(line)