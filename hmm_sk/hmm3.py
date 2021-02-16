#!/usr/bin/env python3

"""
Author: Basem Shaker
Assignment: HMM3
"""

import sys
import argparse
from math import log as log
import numpy as np

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


def dotProduct(vector1, vector2):
    return sum(x*y for x,y in zip(vector1,vector2))

def dotProduct2(vector1, vector2):
    rows = len(vector1)
    columns = len(vector2[0])
    result = [0] * rows * columns
    result = matrixCreator(result,rows,columns)

    for r in range(rows):
        for c in range(columns):
            total = 0
            for i in range(columns):
                total += vector1[r][i] * vector2[i][c]       
            result[r][c] = total  
    return result

def vectorMultiply(vector1,vector2):
    result = []
    if(type(vector2) == float):
        for x in vector1:
            result.append(x*vector2)
    elif(type(vector1) == float):
        for x in vector2:
            result.append(x*vector1)
    else:
        for x,y in zip(vector1,vector2):
            result.append(x*y) 
    return result
    

def vectorDivide(vector1,vector2):
    result = []
    if(type(vector2) == float):
        for x in vector1:
            result.append(x/vector2)
    elif(type(vector1) == float):
        for x in vector2:
            result.append(x/vector1)
    else:
        for x,y in zip(vector1,vector2):
            result.append(x/y) 
    return result


def matrixLog(vector):

    for i in range(len(vector)):
        if(vector[i])!=0.0:
            vector[i]= log(vector[i])
        else:
            vector[i]= float('-inf')
    return vector

def matrixAddition(vector1,vector2):

    result = []
    if(type(vector2) == float):
        for x in vector1:
            result.append(x+vector2)

    else:
        for x,y in zip(vector1,vector2):
            result.append(x+y) 
    return result

def alphaPass(emissionSequence, transMatrix, emissionMatrix, initialState):
    rows = len(emissionSequence[0])
    columns = len(transMatrix)
    alphaMatrix = [0] * rows * columns
    alphaMatrix = matrixCreator(alphaMatrix,rows,columns)
    c0 = 0

    alpha0 = vectorMultiply(initialState[0] , list(zip(*emissionMatrix))[emissionSequence[0][0]])

    for i in range(columns):
        alphaMatrix[0][i]= alpha0[i]
        c0 += alpha0[i]

    c0 = 1/c0
    for i in range(columns):
        alphaMatrix[0][i]*=c0


    for t in range(1, rows):
        ct = 0
        for j in range(columns):

            dotProd =  dotProduct(alphaMatrix[t - 1],list(zip(*transMatrix))[j])
            alphaMatrix[t][j] = dotProd * emissionMatrix[emissionSequence[0][t]][j]
            ct += alphaMatrix[t][j]
        ct = 1/ct
        for i in range(len(alphaMatrix[0])):
            alphaMatrix[t][i]*=ct

    ctminus1 = ct
    return alphaMatrix, ctminus1

def betaPass(emissionSequence, transMatrix, emissionMatrix,ctminus1):
    rows = len(emissionSequence[0])
    columns = len(transMatrix)
    betaMatrix = ([ctminus1] * (rows-1) * columns) + ([1]* columns)
    betaMatrix = matrixCreator(betaMatrix,rows,columns)

    for t in range(rows - 2, -1, -1):
        for j in range(columns):
            multiplicationStep = vectorMultiply(betaMatrix[t+1],list(zip(*emissionMatrix))[emissionSequence[0][t+1]])
            betaMatrix[t][j] = dotProduct(multiplicationStep , transMatrix[j]) * ctminus1

    return betaMatrix



def bwAlgorithm(emissionSequence, transMatrix, emissionMatrix, initialState,iter=10):

    N = len(transMatrix) #length of the observation sequence
    T = len(emissionSequence[0]) #number of states in the model
    M = len(emissionMatrix[0]) #number of observation symbols
    
    for curr_iter in range(iter):
        alphaMatrix,ctminus1 = alphaPass(emissionSequence,transMatrix,emissionMatrix,initialState)
        betaMatrix  = betaPass(emissionSequence,transMatrix,emissionMatrix,ctminus1)
        di_gamma = ([0] * N * N)* (T)
        di_gamma = matrixCreator3D(di_gamma,N,N,T)
        gamma = [0] * N * (T)
        gamma = matrixCreator(gamma,N,T)
        
        # for t in range(T - 1):

        #     dotCalc1 = dotProduct2([alphaMatrix[t]] , transMatrix)[0]
        #     multCalc = vectorMultiply(dotCalc1,emissionMatrix[emissionSequence[0][t+1]])
        #     denominator = dotProduct(multCalc,betaMatrix[t+1])
        #     for i in range(N):
        #         mult1 = vectorMultiply( alphaMatrix[t][i],transMatrix[i])
        #         mult2 = vectorMultiply( mult1,emissionMatrix[emissionSequence[0][t+1]])        
        #         numerator = vectorMultiply( mult2,betaMatrix[t+1])    
        #         for k in range(len(di_gamma[i])):           
        #             di_gamma[i][k][t] = vectorDivide(numerator, denominator)
        #             # di_gamma[i][k][t] = numerator
        #             gamma[i][t] += sum(di_gamma[i][k][t])

        #gamma and di_gamma calculations
        for t in range (T-1):
            for i in range(N):
                gamma[i][t]= 0
                for j in range(N):
                    scalar = alphaMatrix[t][i] * transMatrix[i][j] * emissionMatrix[emissionSequence[0][t+1]][j] * betaMatrix[t+1][j]
                    di_gamma[i][j][t] = scalar
                    gamma[i][t] += di_gamma[i][j][t]

        
        # special case
        for i in range(N):
            gamma[i][-1] = alphaMatrix[-1][i]
        print(gamma)
        # initial state recalculations
        for i in range (N):
            initialState[0][i] = gamma[i][0]

        # alpha recalculations
        for i in range(N):
            denom= 0
            for t in range(T-1):
                denom += gamma[i][t]
            
            for j in range(N):
                numer = 0
                for t in range (T-1):
                    numer += di_gamma[i][j][t]
                transMatrix[i][j]= numer/denom


        for i in range(N):
            denom = 0
            for t in range(T):
                denom += gamma[i][t]

            for j in range(M):
                numer = 0
                for t in range(T-1):
                    if emissionSequence[0][t]==j:
                        numer += gamma[i][t]
                    emissionMatrix[i][j]= numer/denom

    return transMatrix , emissionMatrix


transMatrix, emissionMatrix, initialState , emissionSequence = parse(sys.argv[1:])
transMatrix , emissionMatrix = bwAlgorithm(emissionSequence, transMatrix, emissionMatrix, initialState)

parsedTrans = outputParser(transMatrix)
parsedEmission = outputParser(emissionMatrix)

print(' '.join(map(str,parsedTrans)))
print(' '.join(map(str,parsedEmission)))

