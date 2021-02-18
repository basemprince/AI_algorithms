#!/usr/bin/env python3

"""
Author: Basem Shaker
Assignment: HMM3
"""

import sys
import argparse
from math import log as log

maxIter = 1000000

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

    return inputArray[0] , inputArray[1] , inputArray[2][0] , inputArray[3][0]

def outputParser(inputMatrix):
    rows = len(inputMatrix)
    columns = len(inputMatrix[0])
    parsedList = [rows,columns]

    for i in range(rows):
        for j in range(columns):
            parsedList.append(inputMatrix[i][j])
    return parsedList

def matrixCreator(array,rows,columns,intCheck=False):
    matrix = []
    for i in range(rows):
        rowArray = []
        for j in range(columns):
            if intCheck:
                rowArray.append(int(array[columns * i + j]))
            else:
                rowArray.append(float(array[columns * i + j]))                
        matrix.append(rowArray)

    return matrix


def matrixCreator3D(array,rows,columns,elements,intCheck=False):
    matrix = []
    for i in range(rows):
        rowArray = []
        for j in range(columns):
            internalArray = []
            for k in range(elements):
                if intCheck:
                    internalArray.append(int(array[columns * i + j+k]))
                else:
                    internalArray.append(float(array[columns * i + j+ k])) 
            rowArray.append(internalArray)               
        matrix.append(rowArray)

    return matrix

def alphaPass(observations, aMatrix, bMatrix, piMatrix):

    N = len(aMatrix) #number of states in the model
    T = len(observations) #length of the observation sequence

    alphaMatrix = [0] * T * N
    alphaMatrix = matrixCreator(alphaMatrix,T,N)
    ct_list = [0]
    
    start_ob = observations[0]

    for i in range (N):     
        alphaMatrix[0][i] = piMatrix[i] * bMatrix[i][start_ob]
        ct_list[0] += alphaMatrix[0][i]

    ct_list[0] = 1/ ct_list[0]

    for i in range (N):
        alphaMatrix[0][i] *= ct_list[0]

    for t in range (1, T):

        ct_list.append(0)

        for i in range (N):
            for j in range (N):
                alphaMatrix[t][i]+= alphaMatrix[t-1][j] * aMatrix[j][i]
            current_ob = observations[t]
            alphaMatrix[t][i]*= bMatrix[i][current_ob]
            ct_list[t] += alphaMatrix[t][i]
            
        ct_list[t] = 1/ct_list[t]

        for i in range (N):
            alphaMatrix[t][i] *= ct_list[t]


    return alphaMatrix , ct_list

def betaPass(observations, aMatrix, bMatrix, ct_list):
    N = len(aMatrix) #number of states in the model
    T = len(observations) #length of the observation sequence

    betaMatrix = ([0] * (T-1) * N) + ([ct_list[T-1]]* N)
    betaMatrix = matrixCreator(betaMatrix,T,N)

    for t in range (T-2 , -1 , -1):
        for i in range (N):
            betaMatrix[t][i]=0
            for j in range (N):
                next_ob = observations[t+1]
                betaMatrix[t][i] += aMatrix[i][j] * bMatrix[j][next_ob] * betaMatrix[t+1][j]
            betaMatrix[t][i]*= ct_list[t]


    return betaMatrix

def bwAlgorithm(observations, aMatrix, bMatrix, piMatrix):

    N = len(aMatrix) #number of states in the model
    T = len(observations) #length of the observation sequence
    M = len(bMatrix[0]) #number of observation symbols

    oldLogProb = float('-inf')
    current_iter = 0
    while current_iter <= maxIter:

        alphaMatrix,ct_list = alphaPass(observations,aMatrix,bMatrix,piMatrix)
        betaMatrix  = betaPass(observations,aMatrix,bMatrix,ct_list)

        di_gamma = [0] * T * N * N
        di_gamma = matrixCreator3D(di_gamma,T,N,N)

        gamma = [0] * T * N
        gamma = matrixCreator(gamma,T,N)

        #gamma and di_gamma calculations
        for t in range (T-1):
            for i in range(N):
                gamma[t][i]= 0
                for j in range(N):
                    next_ob = observations[t+1]
                    di_gamma[t][i][j] = alphaMatrix[t][i] * aMatrix[i][j] * bMatrix[j][next_ob] * betaMatrix[t+1][j]
                    gamma[t][i] += di_gamma[t][i][j]

        
        # special case
        for i in range(N):
            gamma[-1][i] = alphaMatrix[-1][i]

        # initial state recalculations
        for i in range (N):
            piMatrix[i] = gamma[0][i]

        # alpha recalculations
        for i in range(N):
            den= 0
            for t in range(T-1):
                den += gamma[t][i]
            
            for j in range(N):
                num = 0
                for t in range (T-1):
                    num += di_gamma[t][i][j]
                aMatrix[i][j]= num/den

        # beta recalculations
        for i in range(N):
            den = 0
            for t in range(T):
                den += gamma[t][i]

            for j in range(M):
                num = 0
                for t in range(T):
                    if observations[t]==j:
                        num += gamma[t][i]
                bMatrix[i][j]= num/den

        # compute log probability of observation given lamda
        current_iter+=1
        print("\r" + str(current_iter), end="")
        logProb = 0
        for i in range(T):
            logProb +=  log(ct_list[i])
        
        logProb *= -1

        if (logProb>oldLogProb):
            oldLogProb = logProb
        else:
            break

    return aMatrix , bMatrix , piMatrix , current_iter


aMatrix, bMatrix, piMatrix , observations = parse(sys.argv[1:])

aMatrix , bMatrix ,piMatrix , current_iter= bwAlgorithm(observations, aMatrix, bMatrix, piMatrix)

print('iterations: ', current_iter)
print(aMatrix)
print('------------------')
print(bMatrix)
print('------------------')
print(piMatrix)

# parsedTrans = outputParser(aMatrix)
# parsedEmission = outputParser(bMatrix)

# print(' '.join(map(str,parsedTrans)))
# print(' '.join(map(str,parsedEmission)))
