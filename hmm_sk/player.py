#!/usr/bin/env python3

from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
import random
import time
from math import log as log

random.seed(40)

T_cap = int(N_STEPS/12) # cap for observatinos allowed
N_cap = 1 # max number of states in the model
M = N_EMISSIONS #number of observation symbols

max_iterations = 1000000
lowest_prob = 1.0e-08

n_optimize = False
limit_obs = True
type_compare = True
debug = False


class PlayerControllerHMM(PlayerControllerHMMAbstract):

    def init_parameters(self):
        """
        In this function you should initialize the parameters you will need,
        such as the initialization of models, or fishes, among others.
        """

        self.lambdas = [] # lambda model for each fish
        self.fish_observations = [[] for _ in range(N_FISH)]  # [fish_id][step]
        self.revealed_fish = {} # {fish_id : fish_type}
        self.fish_w_types = [[] for _ in range(N_SPECIES)] # which fish have same type [fish_type][fish_id]
        self.bestGuess = {} # {fish_id : fish_type}
        self.guess_sequence = [0,N_FISH*0.25,N_FISH*0.5,N_FISH*0.75,N_FISH-1]
        self.type_comparer = [[] for _ in range(N_SPECIES)] # which fish have same type [fish_type][fish_id]
        self.guess_counter = 0
        self.correct_counter = 0
        self.score = [[0,0] for _ in range(N_SPECIES)]
        self.none_counter = 0
        self.un_encountered = [1] * N_SPECIES
        for _ in range(N_FISH):
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
        
        if limit_obs:
            self.observation_compiler_limited(observations)
        else: 
            self.observation_compiler(observations)

        self.bestGuess = {}

        # if len(self.revealed_fish) / N_FISH > 0.7:
        #     global N_cap       
        #     N_cap = 1

        if step > 3:
            for fish_id, model in enumerate(self.lambdas):
                if model.fish_type == None:
                    continue
                # model.initialize_lambda_model()
                
                if n_optimize:
                    model.bwAlgorithm_starter(self.fish_observations[fish_id])
                else:
                    model.initialize_lambda_model()
                    model.bwAlgorithm(self.fish_observations[fish_id])

            for fish_id, obs in enumerate(self.fish_observations):
                self.type_comparer = [[] for _ in range(N_SPECIES)]
                if fish_id in self.revealed_fish:
                    continue

                highestProb = 0
                mostProbType = 0
                known_fish_id_highest = 0
                for known_fish_id , model in enumerate(self.lambdas):         
                    if not model.taught or model.fish_type == None:
                        continue
                    prob = model.probability_of_sequence(obs)

                    # matrixPrinter(model.aMatrix, 'models aMatrix')
                    # matrixPrinter(model.bMatrix, 'models bMatrix')
                    # matrixPrinter(model.piMatrix, 'models piMatrix')
                    # print('fish_id: ', fish_id, ' , known_fish_id: ', known_fish_id, 'fish_type: ', model.fish_type , ' , probability: ', prob, 'most_prob_type: ', mostProbType)
                    # time.sleep(0.1)
                    
                    if type_compare:
                        self.type_comparer[model.fish_type].append(prob)
                    elif prob > highestProb:
                        highestProb = prob
                        mostProbType = model.fish_type
                        known_fish_id_highest = known_fish_id

                if type_compare:                
                    self.type_comparer = average_of_columns(self.type_comparer)
                    mostProbType = max(range(len(self.type_comparer)), key=self.type_comparer.__getitem__)
                    highestProb = self.type_comparer[mostProbType]
                    self.bestGuess.update({fish_id:[mostProbType,highestProb,fish_id]})
                else:
                    self.bestGuess.update({fish_id:[mostProbType,highestProb,known_fish_id_highest]})
        # print('all the list ' , self.bestGuess)
 

        # if len(self.fish_w_types[0]) >1:
        #     for i in self.fish_w_types[0]:
        #         print('fish #: ' , i, ' , ' ,self.fish_observations[i])

        upcoming_guess = None
        if self.bestGuess != {} and not type_compare:
            chosen_fish = max(self.bestGuess.items(), key=lambda k: k[1][1])
            # del self.bestGuess[chosen_fish[0]]
            if debug:
                print('best_guess: ', chosen_fish , 'T length: ' ,  self.lambdas[chosen_fish[1][2]].T, 'N length: ', self.lambdas[chosen_fish[1][2]].N)
            upcoming_guess = [chosen_fish[0],chosen_fish[1][0],chosen_fish[1][1]]

        if self.bestGuess != {} and type_compare:
            chosen_fish = max(self.bestGuess.items(), key=lambda k: k[1][1])
            # del self.bestGuess[chosen_fish[0]]
            if debug:          
                print('best_guess: ', chosen_fish , 'T length: ' ,  self.lambdas[chosen_fish[1][2]].T, 'N length: ', self.lambdas[chosen_fish[1][2]].N)
            upcoming_guess = [chosen_fish[0],chosen_fish[1][0],chosen_fish[1][1]]

        if self.none_counter > 4:
            global lowest_prob
            lowest_prob*= 1.0e-1

    
        if debug:
            print('percent_accuracy: ', round(self.correct_counter/len(self.revealed_fish) * 100.0,0) if len(self.revealed_fish)!=0 else 0 , ' %')

        if upcoming_guess != None:
            if upcoming_guess[2] > lowest_prob:
                # print(upcoming_guess[2])
                self.none_counter = 0
                return (upcoming_guess[0], upcoming_guess[1])
            else:
                self.none_counter+=1
                return None
        else:    
            return (int(self.guess_sequence[step-1]), random.randint(0, N_SPECIES - 1))

    def observation_compiler(self, observations):
        for fish_id , obs in enumerate(observations):
            self.fish_observations[fish_id].append(obs)

    def observation_compiler_limited(self,observations):
        for fish_id , obs in enumerate(observations):
            self.fish_observations[fish_id].append(obs)     
        if len(self.fish_observations[0])>T_cap:
            self.fish_observations= column_delete(self.fish_observations,0)

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

        if correct:
            self.correct_counter +=1

        self.score[true_type][int(correct)] +=1
        score_board = []
        for i in self.score:
            row_sum = sum(i)
            if row_sum != 0:
                accuracy = int(i[1]/row_sum*100)
            else:
                accuracy = None
            score_board.append(accuracy)
        
        if debug:
            print('the guesss is: ' , correct, 'actual id for fish ' , fish_id , ' is: ', true_type)
            print(score_board)
        self.revealed_fish.update({fish_id:true_type})
        self.fish_w_types[true_type].append(fish_id)
        self.lambdas[fish_id].fish_type = true_type
        self.un_encountered[true_type] = 0
        if debug:
            print(self.un_encountered)
        

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

def column_delete (matrix,column_index):
    if isinstance(matrix[0],list):
        rows= len(matrix)
        for row in range(rows):
            matrix[row].pop(column_index)
    else:
        matrix.pop(column_index)    
    return matrix

def average_of_columns(matrix):
    rows= len(matrix)
    for row in range(rows):
        column_count = len (matrix[row])
        if column_count !=0:
            matrix[row] = sum(matrix[row])/column_count
        else: matrix[row] = 0.0
    return matrix

class LAMBDAMODEL:
    def __init__(self):
        self.fish_type = None
        self.aMatrix = []
        self.bMatrix = []
        self.piMatrix = []

        self.N = N_cap
        self.T = 0


        self.maxIter = max_iterations
        self.taught = False

    def initialize_lambda_model(self):
        self.taught = False
        self.aMatrix = []
        self.bMatrix = []
        self.piMatrix = []

        for _ in range(self.N*self.N):
            self.aMatrix.append( random.triangular( 0.5 /self.N , 1.5/self.N))
        self.aMatrix = matrixCreator(self.aMatrix,self.N,self.N)
        self.aMatrix= self.normalizeList(self.aMatrix)

        for _ in range(self.N*M):
            self.bMatrix.append( random.triangular( 0.5/M, 1.5/M) )
        self.bMatrix = matrixCreator(self.bMatrix,self.N,M)
        self.bMatrix = self.normalizeList(self.bMatrix)
            
        for _ in range(self.N):
            self.piMatrix.append( random.triangular(0.5 /self.N , 1.5/self.N) )
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
        alphaMatrix = [0] * self.T * self.N
        alphaMatrix = matrixCreator(alphaMatrix,self.T,self.N)
        ct_list = [0]
        
        start_ob = observations[0]
        for i in range (self.N):
            alphaMatrix[0][i] = self.piMatrix[i] * self.bMatrix[i][start_ob]
            ct_list[0] += alphaMatrix[0][i]

        # if self.ct_list[0]!=0:
        ct_list[0] = 1/ ct_list[0]

        for i in range (self.N):
            alphaMatrix[0][i] *= ct_list[0]

        for t in range (1, self.T):

            ct_list.append(0)

            for i in range (self.N):
                for j in range (self.N):
                    alphaMatrix[t][i]+= alphaMatrix[t-1][j] * self.aMatrix[j][i]
                current_ob = observations[t]
                alphaMatrix[t][i]*= self.bMatrix[i][current_ob]
                ct_list[t] += alphaMatrix[t][i]

            # if self.ct_list[t]!=0:    
            ct_list[t] = 1/ct_list[t]

            for i in range (self.N):
                alphaMatrix[t][i] *= ct_list[t]

        return alphaMatrix , ct_list


    def betaPass(self,observations,ct_list):
        betaMatrix = ([0] * (self.T-1) * self.N) + ([ct_list[self.T-1]]* self.N)
        betaMatrix = matrixCreator(betaMatrix,self.T,self.N)

        for t in range (self.T-2 , -1 , -1):
            for i in range (self.N):
                betaMatrix[t][i]=0
                for j in range (self.N):
                    next_ob = observations[t+1]
                    betaMatrix[t][i] += self.aMatrix[i][j] * self.bMatrix[j][next_ob] * betaMatrix[t+1][j]
                betaMatrix[t][i]*= ct_list[t]
        return betaMatrix

    def zero_counter (self,matrix):
        zero_count = 0
        rows= len(matrix)
        columns = len (matrix[0])
        for row in range(rows):
            for column in range(columns):
                if matrix[row][column]== 0:
                    zero_count += 1
        return zero_count

    def prune_matrix(self,stored_model):

        least_likely_index = min(range(len(stored_model[2])), key=stored_model[2].__getitem__)

        stored_model[2]= stored_model[2].pop(least_likely_index)
        stored_model[3]= stored_model[3].pop(least_likely_index)
        stored_model[3]= column_delete(stored_model[3],least_likely_index)
        stored_model[4]= stored_model[4].pop(least_likely_index)

    def bwAlgorithm_starter(self,observations):
        model_store = []
        self.T = self.set_T(len(observations))
        for state_count in range (N_cap, 0 , -1):
            self.N = state_count
            self.initialize_lambda_model()
            aMatrix,bMatrix,piMatrix,logProb = self.bwAlgorithm(observations,True)
            zero_count = self.zero_counter(aMatrix)
            nK = (state_count * (state_count-1)) - zero_count
            bic = logProb - ((nK/2)*self.T)
            model_store.append([state_count,bic,piMatrix,aMatrix,bMatrix])
            
        highest_bic = float('-inf')
        best_model = None
        for index, stored in enumerate(model_store):
            if stored[1]>highest_bic:
                highest_bic = stored[1]
                best_model = index
        # print(model_store[best_model])
        self.taught = True
        self.N, self.piMatrix, self.aMatrix,self.bMatrix = model_store[best_model][0], model_store[best_model][2],model_store[best_model][3], model_store[best_model][4]
 


    def bwAlgorithm(self,observations,obtimize=False):
        self.T = self.set_T(len(observations))
        if not obtimize:
            self.N = N_cap
        oldLogProb = float('-inf')
        current_iter = 0
        while current_iter <= self.maxIter:

            alphaMatrix, ct_list = self.alphaPass(observations)
            betaMatrix = self.betaPass(observations,ct_list)

            di_gamma = [0] * self.T * self.N * self.N
            di_gamma = matrixCreator(di_gamma,self.T,self.N,self.N)

            gamma = [0] * self.T * self.N
            gamma = matrixCreator(gamma,self.T,self.N)

            #gamma and di_gamma calculations
            for t in range (self.T-1):
                for i in range(self.N):
                    gamma[t][i]= 0
                    for j in range(self.N):
                        next_ob = observations[t+1]
                        di_gamma[t][i][j] = alphaMatrix[t][i] * self.aMatrix[i][j] * self.bMatrix[j][next_ob] * betaMatrix[t+1][j]
                        gamma[t][i] += di_gamma[t][i][j]
  
            # special case
            for i in range(self.N):
                gamma[-1][i] = alphaMatrix[-1][i]

            # initial state recalculations
            for i in range (self.N):
                self.piMatrix[i] = gamma[0][i]

            # alpha recalculations
            for i in range(self.N):
                den= 0
                for t in range(self.T-1):
                    den += gamma[t][i]
                
                for j in range(self.N):
                    num = 0
                    for t in range (self.T-1):
                        num += di_gamma[t][i][j]
                    self.aMatrix[i][j]= num/den if den !=0 else 0

            # beta recalculations
            for i in range(self.N):
                den = 0
                for t in range(self.T):
                    den += gamma[t][i]

                for j in range(M):
                    num = 0
                    for t in range(self.T):
                        if observations[t]==j:
                            num += gamma[t][i]
                    self.bMatrix[i][j]= num/den if den != 0 else 0

            # compute log probability of observation given lamda
            current_iter+=1
            logProb = 0
            for i in range(self.T):
                logProb +=  log(ct_list[i]) #if self.ct_list[i]!= 0 else float('inf')
            
            logProb *= -1
            
            if (logProb>oldLogProb):
                oldLogProb = logProb
            else:
                break
            self.taught = True
        # print(current_iter)
        return self.aMatrix , self.bMatrix, self.piMatrix, logProb

    def probability_of_sequence(self,observations):
        T = len(observations)
        alphaMatrix = [0] * T * self.N
        alphaMatrix = matrixCreator(alphaMatrix,T,self.N)
        
        start_ob = observations[0]

        for i in range (self.N):
            alphaMatrix[0][i] = self.piMatrix[i] * self.bMatrix[i][start_ob]


        for t in range (1, T):

            for i in range (self.N):
                for j in range (self.N):
                    alphaMatrix[t][i]+= alphaMatrix[t-1][j] * self.aMatrix[j][i]
                current_ob = observations[t]
                alphaMatrix[t][i]*= self.bMatrix[i][current_ob]

        # matrixPrinter(alphaMatrix,'alpha')
        # time.sleep(0.2)
        probOfSeq = 0
        for element in alphaMatrix[-1]:
            probOfSeq+=element
        return probOfSeq