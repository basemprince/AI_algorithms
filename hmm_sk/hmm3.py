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

    alpha1 = vectorMultiply(initialState[0] , list(zip(*emissionMatrix))[emissionSequence[0][0]])
    for i in range(len(alphaMatrix[0])):
        alphaMatrix[0][i]= alpha1[i]

    for t in range(1, len(emissionSequence[0])):
        for j in range(len(transMatrix)):
            dotProd =  dotProduct(alphaMatrix[t - 1],list(zip(*transMatrix))[j])
            alphaMatrix[t][j] = dotProd * emissionMatrix[j][emissionSequence[0][t]]

    return alphaMatrix

def betaPass(emissionSequence, transMatrix, emissionMatrix):
    rows = len(emissionSequence[0])
    columns = len(transMatrix)
    betaMatrix = ([0] * (rows-1) * columns) + ([1]* columns)
    betaMatrix = matrixCreator(betaMatrix,rows,columns)

    for t in range(rows - 2, -1, -1):
        for j in range(columns):
            multiplicationStep = vectorMultiply(betaMatrix[t+1],list(zip(*emissionMatrix))[emissionSequence[0][t+1]])
            betaMatrix[t][j] = dotProduct(multiplicationStep , transMatrix[j])

    return betaMatrix



def bwAlgorithm(emissionSequence, transMatrix, emissionMatrix, initialState,iter=100):

    M = len(transMatrix)
    T = len(emissionSequence[0])

    for curr_iter in range(iter):
        alphaMatrix = alphaPass(emissionSequence,transMatrix,emissionMatrix,initialState)
        betaMatrix  = betaPass(emissionSequence,transMatrix,emissionMatrix)

        xi = ([0] * M * M)* (T-1)
        xi = matrixCreator3D(xi,M,M,T-1)
        
        for t in range(T - 1):
            
            dotCalc1 = dotProduct2([list(list(zip(*alphaMatrix))[t])] , transMatrix)[0]

            b_temp = list(zip(*emissionMatrix))
            b_trans=list((list(zip(*b_temp))[emissionSequence[0][t+1]]))

            multCalc = vectorMultiply(dotCalc1,b_trans)
            denominator = dotProduct(multCalc,betaMatrix[t+1])
            for i in range(M):
                mult1 = vectorMultiply( alphaMatrix[t][i],transMatrix[i])
                mult2 = vectorMultiply( mult1,b_trans)
                numerator = vectorMultiply( mult2,list(zip(*betaMatrix))[t+1])    
                for k in range(len(xi[i])):           
                    xi[i][k][t] = vectorDivide(numerator, denominator)

        
        gamma = sum(xi)
        print(gamma)
    # print(xi)

    return transMatrix , emissionMatrix


transMatrix, emissionMatrix, initialState , emissionSequence = parse(sys.argv[1:])
transMatrix , emissionMatrix = bwAlgorithm(emissionSequence, transMatrix, emissionMatrix, initialState)



# print(' '.join(map(str,likelySeq)))

