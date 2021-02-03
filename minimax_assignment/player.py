#!/usr/bin/env python3
import random
from time import time
from time import sleep
import math
from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR

chosen_model = 'IDS' # 'prune' , 'IDS'
chosen_huristic = 'comb_new' # 'score' , 'close', 'close_simple' , 'combination', 'comb_new'
allow_prune = True
allow_hash_table = True
allow_node_sort = True
illegal_check = True
print_allowed = False
debug = False

m_symbol = {
None: 'None',
0: u'\u25EF',
1: u'\u2191',
2: u'\u2193',
3: u'\u2190',
4: u'\u2192'}


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
        self.move_count =0

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
            start_time = time()
            # Create the root node of the game tree
            node = Node(message=msg, player=0)
            #print(msg)
            # Possible next moves: "stay", "left", "right", "up", "down"
            self.move_count +=1
            best_move , elapsed_time = self.search_best_next_move(
                model=model, initial_tree_node=node, move_count=self.move_count, start_time= start_time)

            # Execute next action
            self.sender({"action": best_move, "search_time": elapsed_time})

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
        subdivisions = self.settings.space_subdivisions
        model = minmax_algorithm(initial_data, subdivisions)

        return model


    def search_best_next_move(self, model, initial_tree_node,move_count,start_time):
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
        

        if chosen_model == 'prune':
            leaf, best_move, huristic= model.minmax_prune(initial_tree_node,start_time)
        if chosen_model == 'IDS':
            leaf, best_move, huristic= model.iterativeDeepening(initial_tree_node,start_time)


        if print_allowed:
            move_set, move_symbols= node_winder(leaf)
            best_possible_move = move_set[0]
            print_center(model,move_count,move_symbols,huristic,best_move)
            model.reset()

        elapsed_time = time()-start_time

        return ACTION_TO_STR[best_move], elapsed_time



def node_winder(node):
    move_set = []
    move_symbols = []
    move_set.insert(0,node.move)
    if node.state.player == 0:
        move_symbols.insert(0,'\x1b[1;31m' + m_symbol[node.move] + '\x1b[0m')
    else:
        move_symbols.insert(0,'\x1b[1;32m' + m_symbol[node.move] + '\x1b[0m')        
    parent_node = node.parent
    while parent_node is not None:
        if parent_node.move is not None:
            move_set.insert(0,parent_node.move)
            if parent_node.state.player == 0:
                move_symbols.insert(0,'\x1b[1;31m' + m_symbol[parent_node.move] + '\x1b[0m')
            else:
                move_symbols.insert(0,'\x1b[1;32m' + m_symbol[parent_node.move] + '\x1b[0m')  
        parent_node = parent_node.parent

    return move_set, move_symbols

def print_center(model,move_count,move_symbols,huristic,best_move):
    print('\n############### move num (', move_count, ') ###############')
    print('# max_depth: ',model.deepest , 'average_depth: ' , round(model.ave_depth,2), 'counter:', model.counter)
    print('# elapsed_time: ', (round(model.elapsed_time* 10**3,0)))
    print('# chosen next move: ', move_symbols[0]) 
    print('# chosen child move:','\x1b[1;32m' +  m_symbol[best_move] + '\x1b[0m') 
    print('# move_set: ', end = '')
    print('  '.join(move_symbols))
    print('# huristic: ' , huristic)
    print('# pruned nodes: ', model.prune_count, ' averge_prune: ', round(model.prune_cum / move_count,0))
    print('#############################################')



class minmax_algorithm:
    def __init__(self, initial_data,subdivisions):
        self.initial_data = initial_data
        self.type = chosen_huristic
        self.y_check = 0
        self.subdivisions = subdivisions
        self.deepest = 0
        self.cumDepth = 0
        self.hash_t = {}
        self.elapsed_time = 0
        self.ave_depth = 0
        self.counter = 0
        self.prune_count = 0
        self.prune_cum = 0

    def reset(self):
        self.cumDepth += self.deepest
        self.prune_cum += self.prune_count
        self.deepest = 0
        self.elapsed_time = 0
        self.prune_count = 0

    def distance_calculator(self, my_hook, op_hook, fish):

        distance_eq = lambda hook, fish, x1: (hook[0]- fish[0] + x1)**2 + (hook[1]- fish[1])**2
        check , scenario = self.check_interference(my_hook,op_hook,fish)

        if check and scenario ==1:
            my_distance = distance_eq(my_hook,fish,self.subdivisions)
        elif check and scenario ==2:
            my_distance = distance_eq(my_hook,fish,-self.subdivisions)
        else:            
            my_distance1 = distance_eq(my_hook,fish,0)
            my_distance2 = distance_eq(my_hook,fish,self.subdivisions)
            my_distance3 = distance_eq(my_hook,fish,-self.subdivisions)
            my_distance = min(my_distance1,my_distance2,my_distance3)

        op_distance = distance_eq(op_hook,fish,0)

        return my_distance, op_distance

    def check_interference(self,my_hook,op_hook,fish):
        if(fish[0] >= op_hook[0] >= my_hook[0]):
            return True , 1
        elif(my_hook[0]>= op_hook[0] >= fish[0]):
            return True , 2
        else:
            return False , 0



    def hursitic(self,node):
        if self.type == 'score':
            caught_check_me = node.state.get_caught()[0]
            caught_check_op = node.state.get_caught()[1]

            my_score = node.state.player_scores[0]
            op_score = node.state.player_scores[1]
            if caught_check_me is not None:
                my_score +=  node.state.fish_scores[caught_check_me]
            if caught_check_op is not None:
                op_score +=  node.state.fish_scores[caught_check_op]

            return my_score - op_score

        if self.type == 'close_simple':
            closest = my_distance = float('inf')
            try:
                for key in node.state.fish_positions:
                    my_distance = (( node.state.hook_positions[0][0] - node.state.fish_positions[key][0])**2 + (node.state.hook_positions[0][1] - node.state.fish_positions[key][1])**2)              
                    if my_distance < closest:
                        closest = my_distance
            except:
                pass
            
            final_score = -closest
            return final_score  


        if self.type == 'close':
            fish_list = node.state.fish_positions
            fish_score = node.state.fish_scores

            caught_check_me = node.state.get_caught()[0]
            caught_check_op = node.state.get_caught()[1]
            my_score = node.state.player_scores[0]
            op_score = node.state.player_scores[1]

            if caught_check_me is not None:
                my_score = node.state.fish_scores[caught_check_me]
            if caught_check_op is not None:
                op_score = node.state.fish_scores[caught_check_op]

            score_diff = my_score - op_score

            my_hook = node.state.hook_positions[0]
            try:
                distance_eq = lambda x: (fish_list[x][0] - my_hook[0])**2 + (fish_list[x][1] - my_hook[1])**2
                closest_key = min(fish_list, key=distance_eq)
                closest = distance_eq(closest_key)
            except:
                closest = float('inf')

            final_score = -closest + (score_diff**2)
            return final_score  

        if self.type == 'comb_new':
            
            fish_list = node.state.fish_positions

            fish_score = node.state.fish_scores
            caught_check_me = node.state.get_caught()[0]
            caught_check_op = node.state.get_caught()[1]

            my_hook = node.state.hook_positions[0]
            op_hook = node.state.hook_positions[1]
            score_diff = node.state.player_scores[0]-node.state.player_scores[1]


            if len(fish_list) == 0 and node.parent.state.player_scores[0] != node.state.player_scores[0]:
                closest = float('-inf')
            else:
                closest = my_distance = float('inf')
                
                for key in fish_list:
                    fish = fish_list[key]
                    fish_s = fish_score[key]
                    if fish_s < 0:
                        continue

                    my_distance, op_distance = self.distance_calculator(my_hook,op_hook,fish)   

                    #check if fish is worth going after, if this fish is the closest, and oponent is not within 1 block
                    if my_distance < closest  and caught_check_op != key:
                        target = key
                        if op_distance > my_distance:
                            closest = my_distance / op_distance if op_distance != 0 else 1
                        else:
                            closest = my_distance   


            try:
                final_score = -closest  + fish_score[target] + score_diff
            except:
                final_score = -closest + score_diff

            return final_score  

        if self.type == 'combination':

            fish_list = node.state.fish_positions

            fish_score = node.state.fish_scores
            caught_check_me = node.state.get_caught()[0]
            caught_check_op = node.state.get_caught()[1]

            my_hook = node.state.hook_positions[0]
            oponent_hook = node.state.hook_positions[1]
            score_diff = node.state.player_scores[0]-node.state.player_scores[1]
            
            self.y_check = 0
            closest = my_distance = float('inf')
            fish_away_count = 0.0
            
            #print(len(fish_list) == 0 and caught_check_me is not None , "  ",  caught_check_me)
            if len(fish_list) == 0 :
                closest = float('-inf')

            for key in fish_list:
                fish = fish_list[key]
                fish_s = fish_score[key]


                my_distance1 = ((my_hook[0] - fish[0])**2 + (my_hook[1] - fish [1])**2)
                my_distance2 = ((my_hook[0] - fish[0] + self.subdivisions)**2 + (my_hook[1] - fish [1])**2)
                my_distance3 = ((my_hook[0] - fish[0] - self.subdivisions)**2 + (my_hook[1] - fish [1])**2)
                my_distance = min(my_distance1,my_distance2,my_distance3)

                
                oponent_distance = ((oponent_hook[0] - fish[0])**2 + (oponent_hook[1] - fish [1])**2)

                #check if fish is worth going after, if this fish is the closest, and oponent is not within 1 block
                if fish_s >= 0  and my_distance < closest  and caught_check_op != key:
                    self.target = key
                    if math.sqrt(oponent_distance) > math.sqrt(my_distance):
                        closest = my_distance / oponent_distance
                    else:
                        closest = my_distance   

                #check how many fish are on other side
                if fish[0] > oponent_hook[0] and oponent_hook[0] > my_hook[0]:
                    fish_away_count += 1
                if fish[1] > my_hook[1]:
                    self.y_check += 1

            percent_away = fish_away_count / len(fish_list) if len(fish_list) !=0 else 1

            #move ship to the other side if remaning fish are there
            if percent_away >= 0.99 and node.move == 3:
                closest = float('-inf')
            try:
                final_score = -closest   + fish_score[self.target] + score_diff
            except:
                final_score = -closest + score_diff

            return final_score  



    def illegal_moves(self,CurrentNode,child):

        illegal = False

        #check if last player movement is not opposite the current suggested movement
        if CurrentNode.parent is not None:
            previous_play = CurrentNode.parent
            if previous_play.parent is not None and previous_play.parent.move is not None:
                previous_play = previous_play.parent.move
                current_play = CurrentNode.move
                if previous_play * current_play == 2 or previous_play * current_play == 12:
                    illigal = True

        if CurrentNode.state.hook_positions[1][0] - CurrentNode.state.hook_positions[0][0] == 1 and child.move == 4:
            illegal = True
        elif CurrentNode.state.hook_positions[0][0] - CurrentNode.state.hook_positions[1][0] == 1 and child.move == 3:
            illegal = True


        return illegal

    def minmax_prune(self,CurrentNode,start_time,alpha=float('-inf'),beta=float('inf')):
        self.elapsed_time = time() - start_time
        next_children = CurrentNode.compute_and_get_children()  
        bestNode = None
        bestPossibleMove = None
        move_symbols =  m_symbol[None]

        if allow_hash_table:
            hash_key = str(CurrentNode.depth) \
            + str(CurrentNode.state.player_scores) \
            + str(CurrentNode.state.hook_positions) \
            + str(CurrentNode.state.fish_positions)
            if hash_key in self.hash_t:
                return self.hash_t[hash_key]

        
        if CurrentNode.depth > self.deepest: self.deepest = CurrentNode.depth


        if next_children is None or CurrentNode.depth ==5:        
            huristic = self.hursitic(CurrentNode)
            #print("hurisitic:" ,huristic, "depth:", CurrentNode.depth, "move:",  ACTION_TO_STR[CurrentNode.move])
            return  CurrentNode, CurrentNode.move, huristic

        else:
            current_player = CurrentNode.state.player
            bestPossible = float('-inf') if current_player == 0 else float ('inf')
            for child in next_children:

                if self.illegal_moves(CurrentNode,child) and illegal_check:
                    continue

                node, m, v = self.minmax_prune(child,start_time,alpha,beta)


                if current_player == 0 and v > bestPossible:  #if max turn
                    bestPossible = v
                    bestPossibleMove = child.move
                    bestNode = node
                    alpha = max(alpha,bestPossible)

                elif current_player == 1 and v < bestPossible: # if min turn
                    bestPossible = v
                    bestPossibleMove = child.move
                    bestNode = node
                    beta = min(beta,bestPossible)

                if beta <= alpha and allow_prune:
                    break

                if debug:
                    current_move_symbols = None
                    best_move_symbols = None
                    if CurrentNode.move is not None:
                        _, current_move_symbols = node_winder(CurrentNode)
                    if bestNode.move is not None:
                        _, best_move_symbols = node_winder(bestNode)
                    print('\n##############\n')
                    print('elapsed_time: ', self.elapsed_time)
                    print('depth: ', CurrentNode.depth)
                    print('currentNode player: ', CurrentNode.state.player)
                    print('CurrentNode move: ', m_symbol[CurrentNode.move])
                    print('move_set: ', end='')
                    try:
                        print('  '.join(current_move_symbols))
                    except:
                        print('None')
                    print('alpha:', alpha, ' beta: ', beta)
                    print('my_location: ', node.state.hook_positions[0], ' op location: ', node.state.hook_positions[1])
                    print('huristic: ' , v , 'node_move:' , m_symbol[m])
                    print('best huristic: ' , bestPossible, 'best possible Move: ', m_symbol[bestPossibleMove])
                    print('best_node: ', end='')
                    print('  '.join(best_move_symbols))
                    print('\n##############\n')
                    sleep(0.5)


            if allow_hash_table:
                self.hash_t[hash_key] = bestNode, bestPossibleMove, bestPossible

            return  bestNode, bestPossibleMove, bestPossible

    def minmax_prune_IDS(self,CurrentNode,start_time,depth,alpha=float('-inf'),beta=float('inf')):
        self.elapsed_time = time() - start_time
        next_children = CurrentNode.compute_and_get_children()
        bestPossibleMove = 0
        bestNode = CurrentNode
        self.counter +=1
        self.ave_depth += CurrentNode.depth

        if allow_hash_table:
            hash_key = str(CurrentNode.depth) \
            + str(CurrentNode.state.player_scores) \
            + str(CurrentNode.state.hook_positions) \
            + str(CurrentNode.state.fish_positions)            
            if hash_key in self.hash_t:
                return self.hash_t[hash_key]

        if CurrentNode.depth > self.deepest: self.deepest = CurrentNode.depth

        if next_children is None or CurrentNode.depth >= depth or self.elapsed_time >= 65*1e-3:
            huristic = self.hursitic(CurrentNode)
            #print("hurisitic:" ,huristic, "depth:", CurrentNode.depth, "move:",  ACTION_TO_STR[CurrentNode.move])
            return  CurrentNode, CurrentNode.move, huristic
        else:
            current_player = CurrentNode.state.player
            bestPossible = float('-inf') if current_player == 0 else float ('inf')

            if allow_node_sort:
                sorting_eq = lambda x: self.hursitic(x)
                next_children = sorted(next_children,key=sorting_eq,reverse=True)

            for child in next_children:
                if self.illegal_moves(CurrentNode,child) and illegal_check:
                    continue

                node, m, v = self.minmax_prune_IDS(child,start_time,depth,alpha,beta)

                if current_player == 0 and v > bestPossible:  #if max turn
                    bestPossible = v
                    bestPossibleMove = child.move
                    bestNode = node
                    alpha = max(alpha,bestPossible)

                elif current_player == 1 and v < bestPossible: # if min turn
                    bestPossible = v
                    bestPossibleMove = child.move
                    bestNode = node
                    beta = min(beta,bestPossible)

                if beta <= alpha and allow_prune:
                    self.prune_count+=1
                    break

                if debug:
                    current_move_symbols = None
                    best_move_symbols = None
                    if CurrentNode.move is not None:
                        _, current_move_symbols = node_winder(CurrentNode)
                    if bestNode.move is not None:
                        _, best_move_symbols = node_winder(bestNode)
                    print('\n##############\n')
                    print('elapsed_time: ', self.elapsed_time)
                    print('depth: ', CurrentNode.depth)
                    print('currentNode player: ', CurrentNode.state.player)
                    print('CurrentNode move: ', m_symbol[CurrentNode.move])
                    print('move_set: ', end='')
                    try:
                        print('  '.join(current_move_symbols))
                    except:
                        print('None')
                    print('alpha:', alpha, ' beta: ', beta)
                    print('my_location: ', node.state.hook_positions[0], ' op location: ', node.state.hook_positions[1])
                    print('huristic: ' , v , 'node_move:' , m_symbol[m])
                    print('best huristic: ' , bestPossible, 'best possible Move: ', m_symbol[bestPossibleMove])
                    print('best_node: ', end='')
                    print('  '.join(best_move_symbols))
                    print('\n##############\n')
                    sleep(5)


            if allow_hash_table:
                self.hash_t[hash_key] = bestNode, bestPossibleMove, bestPossible
            return  bestNode, bestPossibleMove, bestPossible

    def iterativeDeepening (self,CurrentNode,start_time):     
        next_children = CurrentNode.compute_and_get_children()
        allowedDepth = 10
        bestPossible = float('-inf')
        bestPossibleMove = 0
        self.deepest = 0
        bestNode = CurrentNode
        self.ave_depth = 0
        self.counter = 0        
        for depth in range (1,allowedDepth+1):

            self.hash_t.clear()

            if depth > self.deepest: self.deepest = depth

            if allow_node_sort:
                sorting_eq = (lambda x:  self.hursitic(x))
                next_children.sort(key=sorting_eq,reverse=True)

            for child in next_children:
                node, m, v = self.minmax_prune_IDS(child,start_time,depth)
                if v > bestPossible:
                    bestPossible = v
                    bestPossibleMove = child.move
                    bestNode = node

        self.ave_depth=self.ave_depth/self.counter
        return  bestNode, bestPossibleMove, bestPossible