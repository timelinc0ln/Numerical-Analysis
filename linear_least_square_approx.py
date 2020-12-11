x = [-1, -0.7, -0.43, -0.14, -0.14, 0.43, 0.71, 1, 1.29, 1.57, 1.86, 2.14, 2.43, 2.71, 3]
y = [-2.25, -0.77, 0.21, 0.44, 0.64, 0.03, -0.22, -0.84, -1.2, -1.03, -0.37, 0.61, 2.67, 5.04, 8.90]

import numpy as np
import pandas as pd
from numpy.linalg import inv

def best_fit(x, y, m = 1):
    """
    Inputs: 
    x: feature variable vector
    y: response variable vector
    
    Outputs:
    beta: parameter ceofficents 
    """
    # create design matrix
    x1 = np.array(x)[np.newaxis]
    y1 = np.array(y)[np.newaxis]
    col1 = np.ones(len(x))[np.newaxis]
    design_matrix = np.concatenate((col1.transpose(), x1.transpose()), axis = 1)
    # fill the matrix for the specified m
    for i in range(1, m): 
        # skip index 0 because we arleady did it
        power_array = col1 + i
        temp = np.power(x1, power_array)
        design_matrix = np.concatenate((design_matrix, temp.transpose()), axis = 1)
    print("Design Matrix:")
    print(design_matrix)
        

    beta = inv(design_matrix.transpose() @ design_matrix) @ design_matrix.transpose() @ y1.transpose()
    return beta

pcoeff = best_fit(x, y , 2)
print("Parameter coefficents:")
print(pcoeff)

def best_fit1(x, y, m = 1):
    """
    This does the same thing as the other one but removes the print statements for brevity 
    
    Inputs: 
    x: feature variable vector
    y: response variable vector
    
    Outputs:
    beta: parameter ceofficents 
    """
    # create design matrix
    x1 = np.array(x)[np.newaxis]
    y1 = np.array(y)[np.newaxis]
    col1 = np.ones(len(x))[np.newaxis]
    design_matrix = np.concatenate((col1.transpose(), x1.transpose()), axis = 1)
    # fill the matrix for the specified m
    for i in range(1, m): 
        # skip index 0 because we arleady did it
        power_array = col1 + i
        temp = np.power(x1, power_array)
        design_matrix = np.concatenate((design_matrix, temp.transpose()), axis = 1)
    #print("Design Matrix:")
    #print(design_matrix)
        

    beta = inv(design_matrix.transpose() @ design_matrix) @ design_matrix.transpose() @ y1.transpose()
    return beta

def compute_poly(pm, x, m):
    xval = 0
    for i in range(m):
        if i == 0:
            xval += pm[i]
        if i == 1:
            xval += pm[i] * x
        xval += pm[i] * x**(i)
    return xval

def optimal_m(x, y, mval):
    for m in range(mval):
        print("m equals:")
        print(m)
        pm = best_fit1(x, y, m)
        if (len(x) - m) == 0:
            pass
        else:
            lead_coeff = 1/(len(x) - m)
            temp = 0
            for i in range(len(x)):
                pmx = compute_poly(pm, x[i], m)
                temp += (pmx - y[i])**2
            mval = temp * lead_coeff
            print("Output: ")
            print(mval)
            
optimal_m(x, y, 4)
# optimal value of m is approximately 4

import matplotlib.pyplot as plt

plt.plot(x,y)

def best_fit2(x, y, m = 1):
    """
    This does the same thing as the other one but removes the print statements for brevity 
    
    Inputs: 
    x: feature variable vector
    y: response variable vector
    
    Outputs:
    beta: parameter ceofficents 
    """
    # create design matrix
    x1 = np.array(x)[np.newaxis]
    y1 = np.array(y)[np.newaxis]
    col1 = np.ones(len(x))[np.newaxis]
    design_matrix = np.concatenate((col1.transpose(), x1.transpose()), axis = 1)
    # fill the matrix for the specified m
    for i in range(1, m): 
        # skip index 0 because we arleady did it
        power_array = col1 + i
        temp = np.power(x1, power_array)
        design_matrix = np.concatenate((design_matrix, temp.transpose()), axis = 1)
    #print("Design Matrix:")
    #print(design_matrix)
        

    beta = inv(design_matrix.transpose() @ design_matrix) @ design_matrix.transpose() @ y1.transpose()
    return beta

def compute_poly(pm, x, m):
    xval = 0
    for i in range(m):
        if i == 0:
            xval += pm[i]
        if i == 1:
            xval += pm[i] * x
        xval += pm[i] * x**(i)
    return xval

def create_y(x, y, mval):
    yval = []
    pm = best_fit2(x, y, mval)
    for i in range(len(x)):
        pmx = compute_poly(pm, x[i], mval)
        yval.append(pmx[0])
    return yval
    
y1 = create_y(x, y, 4)
plt.plot(x, y1)