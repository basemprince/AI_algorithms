#!/usr/bin/env python3

"""
Author: Basem Shaker
Assignment: HMM2
"""

import sys
import argparse
from math import log as log

def parse(arguments):

    parser = argparse.ArgumentParser(description="hmm2 assignment")
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


def vectorMultiply(vector1,vector2):
    result = []
    if(type(vector2) == float):
        for x in vector1:
            result.append(x*vector2)
    else:
        for x,y in zip(vector1,vector2):
            result.append(x*y) 
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

def viterbiAlg(emissionSequence, transMatrix, emissionMatrix, initialState):
    rows = len(emissionSequence[0]) #observation count
    columns = len(transMatrix) #state count

    omegaMatrix = [0] * rows * columns
    omegaMatrix = matrixCreator(omegaMatrix,rows,columns)

    prevMatrix = [0] * (rows-1) * columns
    prevMatrix = matrixCreator(prevMatrix,rows-1,columns)

    likelySeq = [0] * rows

    # initialize omega 0  : δ0 (𝑖) =logπ'𝑏' 𝑂0 ,𝑖=1,...,𝑁
    omega0 = vectorMultiply(initialState[0] , list(zip(*emissionMatrix))[emissionSequence[0][0]])
    omega0 = matrixLog(omega0)

    for i in range(len(omegaMatrix[0])):
        omegaMatrix[0][i]= omega0[i]


 
    for t in range(1, rows):
        for j in range(columns):
            log1= matrixLog(list(list(zip(*transMatrix))[j]))
            try:
                log2=log(emissionMatrix[j][emissionSequence[0][t]])
            except:
                log2 = float('-inf')
            probability = matrixAddition(matrixAddition(omegaMatrix[t-1],log1),log2)     
            maxIndex = probability.index(max(probability))

            prevMatrix[t-1][j] = maxIndex
            omegaMatrix[t][j] = max(probability)
            # print(probability , 'max ' , maxIndex)
    
    latestState = omegaMatrix[rows-1].index(max(omegaMatrix[rows-1]))
    # print(latestState)

    likelySeq[0] = latestState
 
    index = 1
    for i in range(rows - 2, -1, -1):
        likelySeq[index] = prevMatrix[i][latestState]
        latestState = prevMatrix[i][latestState]
        index += 1

    likelySeq = likelySeq[::-1]

    return likelySeq


transMatrix, emissionMatrix, initialState , emissionSequence = parse(sys.argv[1:])
likelySeq = viterbiAlg(emissionSequence,transMatrix,emissionMatrix,initialState)

print(' '.join(map(str,likelySeq)))

