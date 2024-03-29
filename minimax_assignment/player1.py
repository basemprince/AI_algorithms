#!/usr/bin/env python3
import random
from time import time
import math
from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR


class PlayerControllerHuman(PlayerController):
    def player_loop(self):
        """
        Function that generates the loop of the game. In each iteration
        the human plays through the keyboard and send
        this to the game through the sender. Then it receives an
        update of the game through receiver, with this it computes the
        next movement.
        :return:
        """

        while True:
            # send message to game that you are ready
            msg = self.receiver()
            if msg["game_over"]:
                return


class PlayerControllerMinimax(PlayerController):

    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()

    def player_loop(self):
        """
        Main loop for the minimax next move search.
        :return:
        """

        # Generate game tree object
        first_msg = self.receiver()
        # Initialize your minimax model
        model = self.initialize_model(initial_data=first_msg)

        while True:
            msg = self.receiver()

            # Create the root node of the game tree
            node = Node(message=msg, player=0)

            # Possible next moves: "stay", "left", "right", "up", "down"
            best_move = self.search_best_next_move(
                model=model, initial_tree_node=node)

            # Execute next action
            self.sender({"action": best_move, "search_time": None})

    def initialize_model(self, initial_data):
        """
        Initialize your minimax model 
        :param initial_data: Game data for initializing minimax model
        :type initial_data: dict
        :return: Minimax model
        :rtype: object

        Sample initial data:
        { 'fish0': {'score': 11, 'type': 3}, 
          'fish1': {'score': 2, 'type': 1}, 
          ...
          'fish5': {'score': -10, 'type': 4},
          'game_over': False }

        Please note that the number of fishes and their types is not fixed between test cases.
        """
        model = minmax_algorithm(initial_data)

        
        # EDIT THIS METHOD TO RETURN A MINIMAX MODEL ###
        return model

    def search_best_next_move(self, model, initial_tree_node):
        """
        Use your minimax model to find best possible next move for player 0 (green boat)
        :param model: Minimax model
        :type model: object
        :param initial_tree_node: Initial game tree node 
        :type initial_tree_node: game_tree.Node 
            (see the Node class in game_tree.py for more information!)
        :return: either "stay", "left", "right", "up" or "down"
        :rtype: str
        """

        # EDIT THIS METHOD TO RETURN BEST NEXT POSSIBLE MODE FROM MINIMAX MODEL ###
        
        # NOTE: Don't forget to initialize the children of the current node 
        #       with its compute_and_get_children() method!


        best_possible_move,_= model.minmax_prune(initial_tree_node)

        return ACTION_TO_STR[best_possible_move]


class minmax_algorithm:
    def __init__(self, initial_data):
        self.initial_data = initial_data
        self.next_children = []
        self.bestPossibleMove = 0
        self.counter = 0
        self.type = 'combination'
        self.target = None
        self.last_check = None

    def reset(self):
        self.counter = 0


    def hursitic(self,node):
        if self.type == 'score':
            return node.state.player_scores[0]-node.state.player_scores[1]

        if self.type == 'combination':
            fish_list = node.state.fish_positions
            fish_score = node.state.fish_scores
            caught_check = node.state.get_caught()[1]

            my_hook = node.state.hook_positions[0]
            oponent_hook = node.state.hook_positions[1]

            score_diff = node.state.player_scores[0]-node.state.player_scores[1]

            closest = my_distance = float('inf')

            for key in fish_list:
                fish = fish_list[key]
                fish_s = fish_score[key]
                if fish_s >= 0 and caught_check != key :
                    my_distance = (my_hook[0] - fish[0])**2 + (my_hook[1] - fish [1])**2
                    oponent_distance = math.sqrt((oponent_hook[0] - fish[0])**2 + (oponent_hook[1] - fish [1])**2)
                if my_distance < closest and oponent_distance >=1:
                    self.target = key
                    closest = my_distance


            final_score = -closest + score_diff + fish_score[self.target]
            #final_score = 2
            return final_score  



    def minmax_prune(self,CurrentNode,alpha=float('-inf'),beta=float('inf')):

        self.next_children = CurrentNode.compute_and_get_children()

        if self.next_children == []  or CurrentNode.depth >= 3:
            huristic = self.hursitic(CurrentNode)
            #print("hurisitic:" ,huristic, "depth:", CurrentNode.depth, "move:", CurrentNode.move)
            return  CurrentNode.move, huristic


        else:
            current_player = CurrentNode.state.player
            bestPossible = float('-inf') if current_player == 0 else float ('inf')

            for child in self.next_children:
                m, v = self.minmax_prune(child,alpha,beta)

                if current_player == 0 and v > bestPossible and self.target != None:  #if max turn
                    bestPossible = v
                    self.bestPossibleMove = m
                    alpha = max(alpha,bestPossible)

                elif current_player == 1 and v < bestPossible and self.target != None: # if min turn
                    bestPossible = v
                    self.bestPossibleMove = m
                    beta = min(beta,bestPossible)

                if beta <= alpha:
                    break

            return  self.bestPossibleMove, bestPossible   

        
    def minmax_normal(self,CurrentNode):
        
        self.next_children = CurrentNode.compute_and_get_children()

        if self.next_children == [] or CurrentNode.depth >= 3:
            huristic = self.hursitic(CurrentNode)
            return  CurrentNode.move, huristic
        else:
            current_player = self.next_children[0].state.player
            if current_player == 0:
                bestPossible = float('-inf')
                for child in self.next_children:
                    m, v = self.minmax_normal(child)
                    if v > bestPossible:
                        bestPossible = v
                        self.bestPossibleMove = m
                return  self.bestPossibleMove, bestPossible
                
            else: #current_player == B
                bestPossible = float('inf')
                for child in self.next_children:
                    m, v = self.minmax_normal(child)
                    if v < bestPossible:
                        bestPossible = v
                        self.bestPossibleMove = m
                    
                return self.bestPossibleMove, bestPossible