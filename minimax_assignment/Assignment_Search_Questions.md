# Assignment Search - Questions

## Q1
Describe the possible states, initial state, transition function of the KTH fishing derby game.

* Possible states S -> a list with all possible game states, e.g. one state could be: green boat at posision X1 with hook at position Y1, red boat at position X2 with hook at postiion Y2
* Initial game state s0 -> a special state in S which marks the very first state the game finds itself in when it has been started
* Transition function -> function that takes actions bound by rules and returns a new state


## Q2
Describe the terminal states of the KTH fishing derby game.

* A state from S is a terminal state st when μ(p,st)=[], meaning when there is no possible, legal move for player p availible and the game ends.
* In our algorithm: a terminal state is when the end of the observations have been reached, the maximum depth allowed has been reached, or the elapsed time allowed for the turn has been reached


## Q3
Why is ν(A,s)=Score(Green boat)−Score(Red boat) a good heuristic function for the KTH fishing derby (knowing that A plays the green boat and B plays the red boat)?

* It is not only simple to compute, but also represents a good scale or version of a overall utility γ (score should be a good measurement for the utility), at the given state s
* It fullfils the zero-sum game rule. which means γ(A,s)+γ(B,s) = 0 or equivalently γ(A,s) = −γ(B,s)
* 

## Q4
When does ν best approximate the utility function, and why?

* A hurisitc best appproximates the utility when it can chose the best solution given that there exits several solutions, within a reasonable amount of time
* Some huristics can't give a reasonable solution unless we have gone down the tree deep enough 


## Q5
Can you provide an example of a state s where ν(A,s)>0 and B wins in the following turn? (Hint: recall that fish from different types yield different scores).

* first example: if my heuristic is the difference between scores, then it could be due to not calculating deep enough into the tree to figure out that the current movement where ν(A,s)>0 is not a good one because the next movement would cause it to be ν(A,s)<0

* second example: a bad heuristic where it doesn't properly approximate the utility function. eg. a heuristic that calculates the distance difference to fish for the opponents without keeping the score of fish in mind


## Q6
Will η suffer from the same problem (referred to in Q5) as the evaluation function ν? If so, can you provide with an example? (Hint: note how such a heuristic could be problematic in the game of chess, as shown in Figure 2.2).

* yes. for an example a η(A,s)=|w(s,A)|−|l(s,A)| = 5 - 1 = 4. in this example η(A,s) > 0, however, the lose state could be reached literally the next movement compared to the other 5 wins, which may require more than one movement. η does not consider how likely those wins or loses may occur.


