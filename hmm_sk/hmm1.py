#!/usr/bin/env python3

"""
Author: Basem Shaker
Assignment: HMM1
"""

import sys
import argparse


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


transMatrix, emissionMatrix, initialState , emissionSequence = parse(sys.argv[1:])
alphaMatrix = alphaPass(emissionSequence,transMatrix,emissionMatrix,initialState)

probOfSeq = 0
for element in alphaMatrix[-1]:
    probOfSeq+=element

print(probOfSeq)
