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

    for line in allInput:
        lineToArray = line.split()
        rows = int(lineToArray.pop(0))
        columns = int(lineToArray.pop(0))
        formatedMatrix = matrixCreator(lineToArray,rows,columns)
        inputArray.append(formatedMatrix)

    return inputArray[0] , inputArray[1] , inputArray[2] , inputArray[3]

def matrixCreator(array,rows,columns):
    matrix = []
    for i in range(rows):
        rowArray = []
        for j in range(columns):
            rowArray.append(float(array[columns * i + j]))
        matrix.append(rowArray)

    return matrix

def nextMatrixCalculator(currentMatrix,transMatrix):
    rows = len(currentMatrix)
    columns = len(currentMatrix[0])
    NextMatrix = [0] * rows * columns
    NextMatrix = matrixCreator(NextMatrix,rows,columns)
    for i in range(rows):
        for j in range(columns):
            for k in range(len(transMatrix[j])):
                NextMatrix[i][k]+=(currentMatrix[i][j] * transMatrix[j][k])

    return NextMatrix

def emissionProbCalculator(NextMatrix,emissionMatrix):
    rows = len(NextMatrix)
    columns = len(emissionMatrix[0])
    emissionProbMatrix = [0] * columns
    emissionProbMatrix = matrixCreator(emissionProbMatrix,rows,len(emissionMatrix[0]))
    for i in range(rows):
        for j in range(columns):
            for k in range(len(emissionMatrix)):
                emissionProbMatrix[i][j]+= NextMatrix[i][k] * emissionMatrix[k][j] 
    return emissionProbMatrix


def outputParser(inputMatrix):
    rows = len(inputMatrix)
    columns = len(inputMatrix[0])
    parsedList = [rows,columns]

    for i in range(rows):
        for j in range(columns):
            parsedList.append(inputMatrix[i][j])
    return parsedList

def dotProduct(vector1, vector2):
    return sum(x*y for x,y in zip(vector1,vector2))


def alphaPass(emissionSequence, transMatrix, emissionMatrix, initialState):
    rows = len(transMatrix)
    columns = len(emissionSequence[0])
    alphaMatrix = [0] * rows * columns
    alphaMatrix = matrixCreator(alphaMatrix,rows,columns)
    
    print(alphaMatrix)
    return alphaMatrix


transMatrix, emissionMatrix, initialState , emissionSequence = parse(sys.argv[1:])
alphaMatrix = alphaPass(emissionSequence,transMatrix,emissionMatrix,initialState)



# parsedList = outputParser(emissionProbMatrix)


# print(' '.join(map(str,parsedList)))


# NextMatrix = nextMatrixCalculator(initialState,transMatrix)
# emissionProbMatrix = emissionProbCalculator(NextMatrix,emissionMatrix)

# outF = open("hmm0.ans", "w")
# for element in parsedList:
#   outF.write(str(element))
#   outF.write(" ")
# outF.write("\n")
# outF.close()

# print('emissionMatrix', emissionMatrix)
# print('NextMatrix: ', NextMatrix)
# print('emissionProbMatrix: ', emissionProbMatrix)
# print('transitionMatrix', transMatrix)
# print('emissionMatrix', emissionMatrix)
# print('initialState', initialState)
