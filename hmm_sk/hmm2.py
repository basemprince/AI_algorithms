#!/usr/bin/env python3

"""
Author: Basem Shaker
Assignment: HMM2
"""

import sys
import argparse
from math import log as log


def parse(arguments):

    parser = argparse.ArgumentParser(description="hmm0 assignment")
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
    for i in range(rows):
        rowArray = []
        for j in range(columns):
            if intCheck:
                rowArray.append(int(array[columns * i + j]))
            else:
                rowArray.append(float(array[columns * i + j]))                
        matrix.append(rowArray)

    return matrix


def dotProduct(vector1, vector2):
    return sum(x*y for x,y in zip(vector1,vector2))

def vectorMultiply(vector1,vector2):
    result = []
    if(type(vector2) == float):
        for x in vector1:
            result.append(x*vector2)
    else:
        for x,y in zip(vector1,vector2):
            result.append(x*y) 
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

def matrixLog(vector):
    print(type(vector))
    for i in range(len(vector)):
        if(vector[i])!=0:
            vector[i]= log(vector[i])
    return vector


# V = sequence
# a = transmission
# b = emission probability
def viterbiAlg(emissionSequence, transMatrix, emissionMatrix, initialState):
    rows = len(emissionSequence[0])
    columns = len(transMatrix)
    omegaMatrix = [0] * rows * columns
    omegaMatrix = matrixCreator(omegaMatrix,rows,columns)
    prevMatrix = [0] * (rows-1) * columns
    prevMatrix = matrixCreator(prevMatrix,rows-1,columns)

    omega1 = vectorMultiply(initialState[0] , list(zip(*emissionMatrix))[emissionSequence[0][0]])
    omega1 = matrixLog(omega1)

    for i in range(len(omegaMatrix[0])):
        omegaMatrix[0][i]= omega1[i]

    # for t in range(1, len(emissionSequence[0])):
    #     for j in range(len(transMatrix)):
    #         dotProd =  dotProduct(alphaMatrix[t - 1],list(zip(*transMatrix))[j])
    #         alphaMatrix[t][j] = dotProd * emissionMatrix[j][emissionSequence[0][t]]


    for t in range(1, rows):
        for j in range(columns):
            list(zip(*emissionMatrix))[emissionSequence[0][0]])
            # probability = omegaMatrix[t-1]  + log(emissionMatrix[j][emissionSequence[0][t]])
            
            # probability = omega[t - 1] + np.log(a[:, j]) + np.log(b[j, V[t]])
 
            # # This is our most probable state given previous state at time t (1)
            # prev[t - 1, j] = np.argmax(probability)
 
            # # This is the probability of the most probable state (2)
            # omega[t, j] = np.max(probability)





    return omegaMatrix


transMatrix, emissionMatrix, initialState , emissionSequence = parse(sys.argv[1:])
LikelySeq = viterbiAlg(emissionSequence,transMatrix,emissionMatrix,initialState)

