import pandas as pd
import numpy as np
import sys
import argparse
np.set_printoptions(threshold=sys.maxsize)

def normalize(v):
    norm = np.sum(v)
    v = v/norm
    return v

def parse(arguments):

    parser = argparse.ArgumentParser(description="hmm3 assignment")
    parser.add_argument('input' , help="Input file", nargs='?', type=argparse.FileType(), default = sys.stdin)

    args = parser.parse_args(arguments)

    allInput = args.input.readlines()
    inputArray = []

    allInput[-1]= '1 ' + allInput[-1]

    for i, line in enumerate(allInput):
        lineToArray = line.split()
        rows = int(lineToArray.pop(0))
        columns = int(lineToArray.pop(0))
        if i == len(allInput)-1:
            formatedMatrix = matrixCreator(lineToArray,rows,columns,True)
        else:
            formatedMatrix = matrixCreator(lineToArray,rows,columns)
        inputArray.append(formatedMatrix)

    return inputArray[0] , inputArray[1] , inputArray[2] , inputArray[3]


def matrixCreator(array,rows,columns,intCheck=False):
    matrix = []
    matrix = np.zeros((rows, columns))
    for i in range(rows):
        for j in range(columns):
            if intCheck:
                matrix[i,j]=(int(array[columns * i + j]))
            else:
                matrix[i,j]=(float(array[columns * i + j]))                

    return matrix

def forward(V, a, b, initial_distribution):
    alpha = np.zeros((V.shape[0], a.shape[0]))
    alpha[0, :] = initial_distribution * b[:, V[0]]
    alpha[0, :] /= np.sum(alpha[0,:])
    for t in range(1, V.shape[0]):
        ct = 0
        for j in range(a.shape[0]):
            # Matrix Computation Steps
            #                  ((1x2) . (1x2))      *     (1)
            #                        (1)            *     (1)
            alpha[t, j] = alpha[t - 1].dot(a[:, j]) * b[j, V[t]]
            ct += alpha[t,j]
        alpha[t,:]*= ct


    return alpha
 
 
def backward(V, a, b):
    beta = np.zeros((V.shape[0], a.shape[0]))
 
    # setting beta(T) = 1
    beta[V.shape[0] - 1] = np.ones((a.shape[0]))
 
    # Loop in backward way from T-1 to
    # Due to python indexing the actual loop will be T-2 to 0
    for t in range(V.shape[0] - 2, -1, -1):
        for j in range(a.shape[0]):
            beta[t, j] = (beta[t + 1] * b[:, V[t + 1]]).dot(a[j, :])
    
   
    return beta
 
 
def baum_welch(V, a, b, initial_distribution, n_iter=1):
    M = a.shape[0]
    T = len(V)
 
    for n in range(n_iter):
        alpha = forward(V, a, b, initial_distribution)
        beta = backward(V, a, b)
        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            print(b[:, V[t + 1]].shape)
            denominator = np.dot(np.dot(alpha[t, :].T, a) * b[:, V[t + 1]].T, beta[t + 1, :])

            for i in range(M):
                numerator = alpha[t, i] * a[i, :] * b[:, V[t + 1]].T * beta[t + 1, :].T
                xi[i, :, t] = np.divide(numerator,denominator, out=np.zeros_like(numerator), where=denominator!=0)
        gamma = np.sum(xi, axis=1)
        num1 = np.sum(xi, 2)
        den1 = np.sum(gamma, axis=1).reshape((-1, 1))
        a = np.divide(num1,den1, out=np.zeros_like(num1), where=den1!=0)

        # Add additional T'th element in gamma
        gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))
 
        K = b.shape[1]
        denominator = np.sum(gamma, axis=1)
        for l in range(K):
            b[:, l] = np.sum(gamma[:, V == l], axis=1)
 
        den2 = denominator.reshape((-1, 1))
        b = np.divide(b, den2,out=np.zeros_like(b), where=den2!=0)
 
    return (a, b)

transMatrix, emissionMatrix, initialState , emissionSequence = parse(sys.argv[1:])
emissionSequence = emissionSequence[0].astype('int32') 
 
a, b = baum_welch(emissionSequence, transMatrix, emissionMatrix, initialState)

# print( a , ' , ', b)
# print(viterbi(V, a, b, initial_distribution))