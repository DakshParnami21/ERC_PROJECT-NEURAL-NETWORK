import numpy as np
from operations import *


def test(l_weightMatrices, l_biasMatrices, inputs, outputs):
    correct = 0
    for i in range(inputs.shape[0]): 
        estimate = 0 
        a = fpass(inputs[i:i+1], l_weightMatrices[0], l_biasMatrices[0])
        a = sigmoid(a)
        for j in range(len(l_weightMatrices)-1):
            a = fpass(a, l_weightMatrices[j+1], l_biasMatrices[j+1])
            a = sigmoid(a)
        max = 0
        for j in range(10):
            if a[0][j] > max:
                max = a[0][j]
                estimate = j
        if estimate == outputs[i][0]:
            correct += 1
    print('{:.5f}'.format((correct/10000)*100))