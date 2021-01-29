#!/usr/bin/env python3
import random
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
        model.reset()
        #best_possible_move,_= model.minmax_normal(initial_tree_node)
        best_possible_move,_= model.minmax_prune(initial_tree_node)
        #print(model.counter, "   ", best_possible_move, "   ", _)
        #print(best_possible_move, " ", _)
        #random_move = random.randrange(5)
        #return ACTION_TO_STR[random_move]
        return ACTION_TO_STR[best_possible_move]


class minmax_algorithm:
    def __init__(self, initial_data):
        print("##initializing minmax_model")
        self.initial_data = initial_data
        self.next_children = []
        self.bestPossibleMove = 0
        self.counter = 0
        self.type = 'closest_fish'

    def reset(self):
        self.counter = 0


    def hursitic(self,node):
        if self.type == 'score':
            return node.state.player_scores[0]-node.state.player_scores[1]
        if self.type == 'closest_fish':
            fish_list = node.state.fish_positions
            hook_location = node.state.hook_positions[0]
            closest = float('inf')
            for key in fish_list:
                fish = fish_list[key]
                distance = (hook_location[0] - fish[0])**2 + (hook_location[1] - fish [1])**2
                if distance < closest:
                    closest = distance

            return -closest  




    def minmax_normal(self,CurrentNode):
        self.next_children = CurrentNode.compute_and_get_children()

        if self.next_children == [] or CurrentNode.depth >= 6:
            huristic = self.hursitic(CurrentNode)
            #print("hurisitic:" ,huristic, "depth:", CurrentNode.depth, "move:", CurrentNode.move)
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


    def minmax_prune(self,CurrentNode,alpha=float('-inf'),beta=float('inf')):
        self.next_children = CurrentNode.compute_and_get_children()
        self.counter+=1       
        
        #print(self.counter)
        if self.next_children == [] or CurrentNode.depth >=3:
            huristic = self.hursitic(CurrentNode)
            #print("hurisitic:" ,huristic, "depth:", CurrentNode.depth, "move:", CurrentNode.move)
            return  CurrentNode.move, huristic


        
        else:
            current_player = self.next_children[0].state.player   
            if current_player == 0: 
                bestPossible = float('-inf')
                for child in self.next_children:
                    #print( "m", m)
                    m, v = self.minmax_prune(child,alpha,beta)
                    if v > bestPossible:
                        bestPossible = v
                        self.bestPossibleMove = m
                    alpha = max(alpha,bestPossible)
                    if beta <= alpha:
                        break
                return  self.bestPossibleMove, bestPossible   

            else: #current_player == B
                bestPossible = float('inf')
                for child in self.next_children:
                    #print( "m", m)
                    m, v = self.minmax_prune(child,alpha,beta)
                    if v < bestPossible:
                        bestPossible = v
                        self.bestPossibleMove = m
                    beta = min(beta,bestPossible)
                    if beta <= alpha:
                        break
                return  self.bestPossibleMove, bestPossible   


        


    # def minmax_prune(self,CurrentNode,alpha=float('-inf'),beta=float('inf')):
    #     self.next_children = CurrentNode.compute_and_get_children()
    #     self.counter+=1       
        
    #     #print(self.counter)
    #     if self.next_children == [] or CurrentNode.depth >=9:
    #         huristic = CurrentNode.state.player_scores[0]-CurrentNode.state.player_scores[1]
    #         #print("hurisitic:" ,huristic, "depth:", CurrentNode.depth, "move:", CurrentNode.move)
    #         bestPossible = huristic
    #     elif self.next_children[0].state.player ==0:          
    #         bestPossible = float('-inf')
    #         for child in self.next_children:
    #             m = child.move
    #             #print( "m", m)
    #             _, v = self.minmax_prune(child,alpha,beta)
    #             if v > bestPossible:
    #                 bestPossible = v
    #                 if child.depth == 1:
    #                     print("hi")
    #                     self.bestPossibleMove = m
    #             alpha = max(alpha,bestPossible)
    #             if beta <= alpha:
    #                 break

    #     else: #current_player == B
    #         bestPossible = float('inf')
    #         for child in self.next_children:
    #             m = child.move
    #             #print( "m", m)
    #             _, v = self.minmax_prune(child,alpha,beta)
    #             if v < bestPossible:
    #                 bestPossible = v
    #                 if child.depth ==1:
    #                     print("hi")
    #                     self.bestPossibleMove = m
    #             beta = min(beta,bestPossible)
    #             if beta <= alpha:
    #                 break
    #     #print(bestPossible)
    #     return  self.bestPossibleMove, bestPossible   