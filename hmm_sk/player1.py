#!/usr/bin/env python3

from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
import random
import time
from math import log as log

T_cap = 30 # cap for observatinos allowed
N = 3 #number of states in the model
N_cap = 4
M = N_EMISSIONS #number of observation symbols
max_iterations = 400

merge = False
debug = False

class PlayerControllerHMM(PlayerControllerHMMAbstract):

    def init_parameters(self):
        """
        In this function you should initialize the parameters you will need,
        such as the initialization of models, or fishes, among others.
        """

        self.lambdas = []
        self.fish_observations = [[] for _ in range(N_FISH)]  # [fish_id][step]
        self.revealed_fish = {} # {fish_id : fish_type}
        self.fish_w_types = [[] for _ in range(N_SPECIES)] # which fish have same type [fish_type][fish_id]
        self.bestGuess = {} # {fish_id : fish_type}

        for _ in range(N_SPECIES):
            model = LAMBDAMODEL()
            self.lambdas.append(model)

        
    def guess(self, step, observations):
        """
        This method gets called on every iteration, providing observations.
        Here the player should process and store this information,
        and optionally make a guess by returning a tuple containing the fish index and the guess.
        :param step: iteration number
        :param observations: a list of N_FISH observations, encoded as integers
        :return: None or a tuple (fish_id, fish_type)
        """

        self.observation_compiler(observations)

        self.bestGuess = {}
        if step > 5:
            for fish_type, model in enumerate(self.lambdas):
                
                model.initialize_lambda_model()
                if self.fish_w_types[fish_type] != []:
                    model.taught = True
                    if merge:
                        known_fish_ids = self.fish_w_types[fish_type]
                        all_obs = []
                        for current in known_fish_ids:
                            all_obs += self.fish_observations[current] 
                        model.bwAlgorithm(all_obs)
                    else:
                        known_fish_id = self.fish_w_types[fish_type][0]
                        model.bwAlgorithm(self.fish_observations[known_fish_id])

            for fish_id, obs in enumerate(self.fish_observations):
                if fish_id in self.revealed_fish:
                    continue
                highestProb = 0
                mostProbModel = 0
                for model_id , model in enumerate(self.lambdas):
                    if not model.taught:
                        continue
                    prob = model.probability_of_sequence(obs)
                    # matrixPrinter(model.aMatrix, 'models aMatrix')
                    # matrixPrinter(model.bMatrix, 'models bMatrix')
                    # matrixPrinter(model.piMatrix, 'models piMatrix')
                    # print('fish_id: ', fish_id, ' , fish_type: ', model_id , ' , probability: ', prob, 'most_prob_type: ', mostProbModel)
                    # time.sleep(0.4)
                    if prob > highestProb:
                        highestProb = prob
                        mostProbModel = model_id
                self.bestGuess.update({fish_id:[mostProbModel,highestProb]})
        # print('all the list ' , self.bestGuess)
        

        # if len(self.fish_w_types[0]) >1:
        #     for i in self.fish_w_types[0]:
        #         print('fish #: ' , i, ' , ' ,self.fish_observations[i])

        upcoming_guess = None
        if self.bestGuess != {}:
            chosen_fish = max(self.bestGuess.items(), key=lambda k: k[1][1])
            print('best_guess: ', chosen_fish , 'T length:' ,  self.lambdas[chosen_fish[1][0]].T)
            upcoming_guess = [chosen_fish[0],chosen_fish[1][0], chosen_fish[1][1]]

        
        if upcoming_guess != None:
            # print('guess porb: ', upcoming_guess[2])
            return (upcoming_guess[0], upcoming_guess[1])
        else:
            return (step % N_FISH, random.randint(0, N_SPECIES - 1))

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
        self.revealed_fish.update({fish_id:true_type})
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

def matrixPrinter(matrix,matrix_name):
    
    if isinstance(matrix[0],list):
        rows= len(matrix)
        columns = len (matrix[0])
        print('#### the ' , matrix_name , ' matrix ( ' , rows , 'X', columns, ') ####' )
        for row in range(rows):
            print(matrix[row])
    else:
        rows = 1
        columns = len (matrix)
        print('#### the ' , matrix_name , ' matrix ( ' , rows , 'X', columns, ') ####' )
        print(matrix)


class LAMBDAMODEL:
    def __init__(self):
        self.aMatrix = []
        self.bMatrix = []
        self.piMatrix = []

        self.T = 0

        self.alphaMatrix = []
        self.betaMatrix = []
        self.ct_list = []
        self.maxIter = max_iterations
        self.taught = False

    def initialize_lambda_model(self):
        self.taught = False
        self.aMatrix = []
        self.bMatrix = []
        self.piMatrix = []

        for _ in range(N*N):
            self.aMatrix.append( random.triangular( 0.5 /N , 1.5/N, 1/N ))
        self.aMatrix = matrixCreator(self.aMatrix,N,N)
        self.aMatrix= self.normalizeList(self.aMatrix)

        for _ in range(N*M):
            self.bMatrix.append( random.triangular( 0.5/M, 1.5/M, 1/M) )
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

    def set_T(self,obs_len):
        return obs_len if obs_len <=T_cap else T_cap

    def alphaPass(self,observations):
        self.alphaMatrix = [0] * self.T * N
        self.alphaMatrix = matrixCreator(self.alphaMatrix,self.T,N)
        self.ct_list = [0]
        
        start_ob = observations[0]

        for i in range (N):
            self.alphaMatrix[0][i] = self.piMatrix[i] * self.bMatrix[i][start_ob]
            self.ct_list[0] += self.alphaMatrix[0][i]

        self.ct_list[0] = 1/ self.ct_list[0]

        for i in range (N):
            self.alphaMatrix[0][i] *= self.ct_list[0]

        for t in range (1, self.T):

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
        self.betaMatrix = ([0] * (self.T-1) * N) + ([self.ct_list[self.T-1]]* N)
        self.betaMatrix = matrixCreator(self.betaMatrix,self.T,N)

        for t in range (self.T-2 , -1 , -1):
            for i in range (N):
                self.betaMatrix[t][i]=0
                for j in range (N):
                    next_ob = observations[t+1]
                    self.betaMatrix[t][i] += self.aMatrix[i][j] * self.bMatrix[j][next_ob] * self.betaMatrix[t+1][j]
                self.betaMatrix[t][i]*= self.ct_list[t]


    def bwAlgorithm_starter(self,observatinos):
        for state_count in range (1,N_cap):
            pass

    def bwAlgorithm(self,observations):
        self.T = self.set_T(len(observations))
        oldLogProb = float('-inf')
        current_iter = 0
        while current_iter <= self.maxIter:

            self.alphaPass(observations)
            self.betaPass(observations)

            di_gamma = [0] * self.T * N * N
            di_gamma = matrixCreator(di_gamma,self.T,N,N)

            gamma = [0] * self.T * N
            gamma = matrixCreator(gamma,self.T,N)

            #gamma and di_gamma calculations
            for t in range (self.T-1):
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
                for t in range(self.T-1):
                    den += gamma[t][i]
                
                for j in range(N):
                    num = 0
                    for t in range (self.T-1):
                        num += di_gamma[t][i][j]
                    self.aMatrix[i][j]= num/den

            # beta recalculations
            for i in range(N):
                den = 0
                for t in range(self.T):
                    den += gamma[t][i]

                for j in range(M):
                    num = 0
                    for t in range(self.T):
                        if observations[t]==j:
                            num += gamma[t][i]
                    self.bMatrix[i][j]= num/den

            # compute log probability of observation given lamda
            current_iter+=1
            logProb = 0
            for i in range(self.T):
                logProb +=  log(self.ct_list[i])
            
            logProb *= -1
            # print(current_iter)
            if (logProb>oldLogProb):
                oldLogProb = logProb
            else:
                break

    def probability_of_sequence(self,observations):
        T = len(observations)
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

        # matrixPrinter(alphaMatrix,'alpha')
        # time.sleep(0.2)
        probOfSeq = 0
        for element in alphaMatrix[-1]:
            probOfSeq+=element
        return probOfSeq