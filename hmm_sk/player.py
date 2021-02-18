#!/usr/bin/env python3

from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
import random
import time
from math import log as log

N = 4 #number of states in the model
T = 0 #length of the observation sequence
M = N_EMISSIONS #number of observation symbols
max_iterations = 100

debug = False

class PlayerControllerHMM(PlayerControllerHMMAbstract):

    def init_parameters(self):
        """
        In this function you should initialize the parameters you will need,
        such as the initialization of models, or fishes, among others.
        """

        self.lambdas = []
        self.fish_observations = [[] for _ in range(N_FISH)]  # [fish_id][step]
        self.fish_predictions = [None] * N_FISH # None for unknown type
        self.revealed_fish = [None] * N_FISH
        self.fish_w_types = [[] for _ in range(N_SPECIES)] # which fish have same type [fish_type][fish_id]

        for _ in range(N_SPECIES):
            model = LAMBDAMODEL()
            self.lambdas.append(model)

        # if debug:
        #     for i, model in enumerate(self.lambdas):
        #         print('############# model # ', i, '###############')
        #         print('#aMatrix: ', model.aMatrix)
        #         print('#bMatrix: ', model.bMatrix)
        #         print('#piMatrix: ', model.piMatrix)
        
    def guess(self, step, observations):
        """
        This method gets called on every iteration, providing observations.
        Here the player should process and store this information,
        and optionally make a guess by returning a tuple containing the fish index and the guess.
        :param step: iteration number
        :param observations: a list of N_FISH observations, encoded as integers
        :return: None or a tuple (fish_id, fish_type)
        """
        global T
        T = step
        self.observation_compiler(observations)

        if step > 10:
            for model in self.lambdas:
                model.initialize_lambda_model()
                model.bwAlgorithm(self.fish_observations[0])
                if debug:
                    for i, model in enumerate(self.lambdas):
                        print('############# model # ', i, '###############')
                        print('#alphaMatrix: ', model.alphaMatrix)
                        print('#betaMatrix: ', model.betaMatrix)
                        print('#piMatrix: ', model.piMatrix)

            for fish_id,obs in enumerate(self.fish_observations):
                highestProb = 0
                for model_id , model in enumerate(self.lambdas):
                    prob = model.probability_of_sequence(obs)
                    if prob > 0:
                        highestProb = model_id


        # This code would make a random guess on each step:
        return (step % N_FISH, random.randint(0, N_SPECIES - 1))

        # return None

    def observation_compiler(self, observations):
        for fish_id , obs in enumerate(observations):
            self.fish_observations[fish_id].append(obs)

    def reveal(self, correct, fish_id, true_type):
        """
        This methods gets called whenever a guess was made.
        It informs the player about the guess result
        and reveals the correct type of that fish.
        :param correct: tells if the guess was correct
        :param fish_id: fish's index
        :param true_type: the correct type of the fish
        :return:
        """
        self.revealed_fish[fish_id] = true_type
        self.fish_w_types[true_type].append(fish_id)
        

def matrixCreator(array,rows,columns,elements=0,intCheck=False):
    matrix = []
    for i in range(rows):
        rowArray = []
        for j in range(columns):
            internalArray = []
            if elements == 0:
                if intCheck:
                    rowArray.append(int(array[columns * i + j]))
                else:
                    rowArray.append(float(array[columns * i + j])) 
            else:
                for k in range(elements):
                    if intCheck:
                        internalArray.append(int(array[columns * i + j+k]))
                    else:
                        internalArray.append(float(array[columns * i + j+ k])) 
                rowArray.append(internalArray)               
        matrix.append(rowArray)

    return matrix

class LAMBDAMODEL:
    def __init__(self):
        self.aMatrix = []
        self.bMatrix = []
        self.piMatrix = []

        self.alphaMatrix = []
        self.betaMatrix = []
        self.ct_list = []
        self.maxIter = max_iterations
    
    def initialize_lambda_model(self):
        self.aMatrix = []
        self.bMatrix = []
        self.piMatrix = []

        for _ in range(N*N):
            self.aMatrix.append( random.triangular( 0.5 /N , 1.5/N, 1/N ))
        self.aMatrix = matrixCreator(self.aMatrix,N,N)
        self.aMatrix= self.normalizeList(self.aMatrix)

        for _ in range(N*M):
            self.bMatrix.append( random.triangular( 0.5/M, 1.5/M, 1/N) )
        self.bMatrix = matrixCreator(self.bMatrix,N,M)
        self.bMatrix = self.normalizeList(self.bMatrix)
            
        for _ in range(N):
            self.piMatrix.append( random.triangular(0.5 /N , 1.5/N, 1/N) )
        self.piMatrix = self.normalizeList(self.piMatrix)
    
    def normalizeList(self,matrix):
        if isinstance(matrix[0], list):
            normalized = []
            for row in matrix:
                normalized.append([float(x)/sum(row) for x in row])
            return normalized
        else:
            return [float(x)/sum(matrix) for x in matrix]


    def alphaPass(self,observations):
        self.alphaMatrix = [0] * T * N
        self.alphaMatrix = matrixCreator(self.alphaMatrix,T,N)
        self.ct_list = [0]
        
        start_ob = observations[0]

        for i in range (N):
            self.alphaMatrix[0][i] = self.piMatrix[i] * self.bMatrix[i][start_ob]
            self.ct_list[0] += self.alphaMatrix[0][i]

        self.ct_list[0] = 1/ self.ct_list[0]

        for i in range (N):
            self.alphaMatrix[0][i] *= self.ct_list[0]

        for t in range (1, T):

            self.ct_list.append(0)

            for i in range (N):
                for j in range (N):
                    self.alphaMatrix[t][i]+= self.alphaMatrix[t-1][j] * self.aMatrix[j][i]
                current_ob = observations[t]
                self.alphaMatrix[t][i]*= self.bMatrix[i][current_ob]
                self.ct_list[t] += self.alphaMatrix[t][i]
                
            self.ct_list[t] = 1/self.ct_list[t]

            for i in range (N):
                self.alphaMatrix[t][i] *= self.ct_list[t]


    def betaPass(self,observations):

        self.betaMatrix = ([0] * (T-1) * N) + ([self.ct_list[T-1]]* N)
        self.betaMatrix = matrixCreator(self.betaMatrix,T,N)

        for t in range (T-2 , -1 , -1):
            for i in range (N):
                self.betaMatrix[t][i]=0
                for j in range (N):
                    next_ob = observations[t+1]
                    self.betaMatrix[t][i] += self.aMatrix[i][j] * self.bMatrix[j][next_ob] * self.betaMatrix[t+1][j]
                self.betaMatrix[t][i]*= self.ct_list[t]

    def bwAlgorithm(self,observations):

        oldLogProb = float('-inf')
        current_iter = 0
        while current_iter <= self.maxIter:

            self.alphaPass(observations)
            self.betaPass(observations)

            di_gamma = [0] * T * N * N
            di_gamma = matrixCreator(di_gamma,T,N,N)

            gamma = [0] * T * N
            gamma = matrixCreator(gamma,T,N)

            #gamma and di_gamma calculations
            for t in range (T-1):
                for i in range(N):
                    gamma[t][i]= 0
                    for j in range(N):
                        next_ob = observations[t+1]
                        di_gamma[t][i][j] = self.alphaMatrix[t][i] * self.aMatrix[i][j] * self.bMatrix[j][next_ob] * self.betaMatrix[t+1][j]
                        gamma[t][i] += di_gamma[t][i][j]

            
            # special case
            for i in range(N):
                gamma[-1][i] = self.alphaMatrix[-1][i]

            # initial state recalculations
            for i in range (N):
                self.piMatrix[i] = gamma[0][i]

            # alpha recalculations
            for i in range(N):
                den= 0
                for t in range(T-1):
                    den += gamma[t][i]
                
                for j in range(N):
                    num = 0
                    for t in range (T-1):
                        num += di_gamma[t][i][j]
                    self.aMatrix[i][j]= num/den

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
                    self.bMatrix[i][j]= num/den

            # compute log probability of observation given lamda
            current_iter+=1
            logProb = 0
            for i in range(T):
                logProb +=  log(self.ct_list[i])
            
            logProb *= -1

            if (logProb>oldLogProb):
                oldLogProb = logProb
            else:
                break
    def probability_of_sequence(self,observations):

        alphaMatrix = [0] * T * N
        alphaMatrix = matrixCreator(alphaMatrix,T,N)
        
        start_ob = observations[0]

        for i in range (N):
            alphaMatrix[0][i] = self.piMatrix[i] * self.bMatrix[i][start_ob]


        for t in range (1, T):

            for i in range (N):
                for j in range (N):
                    alphaMatrix[t][i]+= alphaMatrix[t-1][j] * self.aMatrix[j][i]
                current_ob = observations[t]
                alphaMatrix[t][i]*= self.bMatrix[i][current_ob]

        probOfSeq = 0
        for element in alphaMatrix[-1]:
            probOfSeq+=element
        return probOfSeq