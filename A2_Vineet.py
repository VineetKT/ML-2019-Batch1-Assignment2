import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

if __name__ == '__main__':
    ## for the White Wine dataset
    data_White = pd.read_csv('winequality-white.csv')       ## for the White Wine dataset
    array = np.array(data_White)

    '''
    ## for the Red Wine dataset
    data_Red = pd.read_csv('winequality-red.csv')
    array = np.array(data_Red)
    '''
    
    min_Y = array[:,11].min()
    max_Y = array[:,11].max()

    for i in range(array.shape[1]):
        array[:,i] = (array[:,i] - array[:,i].min())/(array[:,i].max() - array[:,i].min())

    seed = 0.9   # fraction of dataset taken for training of model
    d = int(seed*len(array))
    X_train = array[:d, 0:-1]
    X1 = np.ones([len(X_train),1])
    X_train = np.concatenate((X1, X_train), axis = 1)
    dim = X_train.shape
    m = dim[0]  # no. of instances
    n = dim[1] # no. of features

    Y_train = array[:d,-1]
    Y_train = Y_train.reshape(-1,1)

    theta = np.zeros(n)[np.newaxis].transpose()
    alpha = 1           # Hyperparameter (learning rate)

    #cost function
    temp = (np.matmul(X_train,theta) - Y_train)
    J_old = (np.matmul(temp.transpose(), temp))/(2*m)
    cost1 = []

    iter1 = 10000
    for i in range(iter1):
        temp = (np.matmul(X_train,theta) - Y_train)
        theta = theta - (alpha*np.matmul( X_train.transpose(),temp))/m
        temp = (np.matmul(X_train,theta) - Y_train)
        J_new = (np.matmul(temp.transpose(), temp))/(2*m)

        if abs(J_new - J_old) <= 0.0001:
            break

        J_old = J_new
        cost1.append(J_new[0][0])
    print('Number of iterations taken: ', i)
    print('Theta vector: \n', theta)
    print('Cost function value in each iteration: \n',list(cost1))
    print(list(cost1))

    ## Validation 
    X_valid = array[d:, 0:-1]
    X1 = np.ones([len(X_valid),1])
    X_valid = np.concatenate((X1, X_valid), axis = 1)

    
    Y_valid = array[d:,-1]
    Y_valid = min_Y + (max_Y - min_Y)*Y_valid.reshape(-1,1)
    Y_predicted = min_Y + (max_Y - min_Y)*np.matmul(X_valid, theta)

    ## SSE: Error Sum of Squares
    error = Y_valid - Y_predicted
    len_ = len(error)
    error_T = error.reshape(1,len_)
    SSE = float(np.matmul(error_T , error))
    print('Error Sum of Squares (SSE): ', SSE)
    print('Mean squared sum of errors (MSE): ', SSE/len_)
    print('Root Mean Squared sum of Errors (RMSE): ', math.sqrt(SSE)/len_)
    print('Mean absolute error(MAE): ', float(sum(abs(Y_valid-Y_predicted))/len_))
    
    M = 100/len_*sum(abs((Y_valid-Y_predicted)/Y_valid))
    print('Mean Absolute Percentage Error (MAPE): ', M)

    ## SSR: Regression Sum of Squares
    err = Y_predicted - Y_valid.mean()
    SSR = float(np.matmul(err.reshape(1, len_), err))
    print('Regression Sum of Squares: ', SSR)

    SSTO = SSR + SSE   # SSTO: total sum of squares
    R_sqr = SSR/SSTO
    print('R-squared Value: ', R_sqr)
    
    plt.plot(cost1)
    plt.xlabel('iterations')
    plt.ylabel('Loss function')
    plt.title('Loss function Vs. iterations')
    plt.show()
