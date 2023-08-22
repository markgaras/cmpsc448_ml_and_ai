import random
import numpy as np
import matplotlib.pyplot as plt
from Problem5 import bgd_l2, sgd_l2

if __name__ == '__main__':
    # Put the code for the plots here, you can use different functions for each
    # part

    # Gradient Descent Initialization
    w = np.random.random(2)    # Generate 2 random numbers from 0-2
    data = np.load('data.npy') # Load in data to use
    x = np.hsplit(data, 2)     # Split data through middle
    y = x[1]                   # y is the second half (or outputs)
    x = x[0]                   # x is the first half (or inputs)
    
    def plotGraphGD(data, y, w, eta, delta, lam, num_iter, numTest): 
        w2, fwHistory = bgd_l2(data, y, w, eta, delta, lam, num_iter)
        plt.plot(fwHistory)
        plt.title("History of Objective Function using GD, Number: " + numTest)
        plt.xlabel("Iteration Number")
        plt.ylabel("Objective Function")
        plt.show()
    
    # The GD tests
    plotGraphGD(x, y, w, .05, .1, .001, 50, "1")
    plotGraphGD(x, y, w, .1, .01, .001, 50, "2")
    plotGraphGD(x, y, w, .1, 0, .001, 100, "3")
    plotGraphGD(x, y, w, .1, 0, 0, 100, "4")
 
    def plotGraphSGD(data, y, w, eta, delta, lam, num_iter, numTest):
        w2, fwHistory = sgd_l2(data, y, w, eta, delta, lam, num_iter, i=-1)
        plt.plot(fwHistory)
        plt.title("History of Objective Function using SGD, Number: " + numTest)
        plt.xlabel("Iteration Number")
        plt.ylabel("Objective Function")
        plt.show()

    # The SGD tests
    plotGraphSGD(x, y, w, 1, .1, .5, 800, "1")
    plotGraphSGD(x, y, w, 1, .01, .1, 800, "2")
    plotGraphSGD(x, y, w, 1, 0, 0, 40, "3")
    plotGraphSGD(x, y, w, 1, 0, 0, 800, "4")