import math
import random
import numpy as np

def getGradient(data, y, w, eta, delta, lam, num_iter, wt, x):
    gradient = 0
    for j in range(len(x)):
        if (y[j] >= (np.dot(wt, x[j]) + delta)):
            gradient += 2 * (y[j] - np.dot(wt, x[j]) - delta) * -x[j]
        elif (abs(y[j] - np.dot(wt, x[j])) < delta):
            gradent += 0;
        elif (y[j] <= (np.dot(wt, x[j]) - delta)):
            gradient += 2 * (y[j] - np.dot(wt, x[j]) + delta) * -x[j]
        gradient = gradient / len(x)
        gradient += 2 * lam * sum(wt)
        return gradient
    
def getFw(data, y, w, eta, delta, lam, num_iter, wt, x): # Returns f(w)
    fw = 0
    for t in range(len(x)):
        if (y[t] >= (np.dot(wt, x[t]) + delta)):
            fw += ((y[t] - np.dot(wt, x[t]) - delta) ** 2)
        elif (abs(y[t] - np.dot(wt, x[t])) < delta):
            fw += 0
        elif (y[t] <= (np.dot(wt, x[t]) - delta)):
            fw += ((y[t] - np.dot(wt, x[t]) + delta) ** 2)
        fw = fw/len(x)
        fw += lam * sum(wt**2)
    return fw

def bgd_l2(data, y, w, eta, delta, lam, num_iter):
    uno = np.full((100, 1), 1) # Creating a matrix of just ones
    x = np.concatenate((uno, data), axis = 1) # Adding it to the front of the data matrix
    w2 = w
    fwHistory = []
    for i in range(num_iter):                                     
        wt = np.transpose(w2)
        
        # Getting the gradient, using respective derivatives from the piecewise function
        gradient = getGradient(data, y, w, eta, delta, lam, num_iter, wt, x)
        
        w2 = w2 - (eta * gradient)  # How to change w2 according to GD
        wt = np.transpose(w2)
        
        fw = getFw(data, y, w, eta, delta, lam, num_iter, wt, x)
        fwHistory.append(fw)
        
    return w2, fwHistory

def sgd_l2(data, y, w, eta, delta, lam, num_iter, i=-1):
    uno = np.full((100, 1), 1)                # Creating a matrix of just ones
    x = np.concatenate((uno, data), axis = 1) # Adding it to the front of the data matrix
    w2 = w
    fwHistory = []
    
    if (i != -1):
        numer_iter = 1
    else:
        i = random.randrange(0, len(x))
        
    for j in range(1, num_iter + 1):
        wt = np.transpose(w2)
        # Getting the gradient, using respective derivatives from the piecewise function
        gradient = getGradient(data, y, w, eta, delta, lam, num_iter, wt, x)
        
        w2 = w2 - ((eta / math.sqrt(j)) * gradient) # How to change w2 according to SGD
        
        wt = np.transpose(w2)
        fw = getFw(data, y, w, eta, delta, lam, num_iter, wt, x)
        fwHistory.append(fw)
        i = random.randrange(0, len(x))
        
    return w2, fwHistory