# Complete this class for all parts of the project

from pacman_module.game import Agent
from pacman_module.pacman import Directions
from random import randint
import random
import math
from collections import Counter

#import util

class PacmanAgent(Agent):
    def __init__(self, args):
        """
        Arguments:
        ----------
        - `args`: Namespace of arguments from command-line prompt.
        """
        self.args = args

        self.alpha = float(0.2)
        self.epsilon = float(0.05)
        self.gamma = float(0.8)

        # Q values
        #self.Q_values = dict() # or util.Counter()
        self.Q_values = Counter()

        # current score
        self.score = 0

        self.training_num = int(20)
        # number of games we have played
        self.episodes_total = 0

        # last state
        self.last_state = []
        # last action
        self.last_action = []

    def coinFlip(self, p):
        """
        Simulates flipping a coin with probability p of returning True.
        
        Args:
        p (float): The probability of returning True.
        
        Returns:
        bool: True with probability p, False otherwise.
        """
        return (random.random() < p)
    
    def all_values_same(self, counter):
        """
        Check if all values in the Counter are the same.
        
        Args:
        counter (Counter): A Counter object with actions as keys and Q-values as values.
        
        Returns:
        bool: True if all values are the same, False otherwise.
        """
        if not counter:
            return True
        
        first_value = next(iter(counter.values()))
        return all(value == first_value for value in counter.values())


    # returns the Q value for a given state
    def getQ_Value(self, state, action):
        return self.Q_values[(state,action)]
    
    # return the maximum Q values of each state
    def getMaxQ_Values(self, state):
        Q_list = []
        #for action in state.getLegalPacmanActions():
        for action in state.getLegalActions():
            Q = self.getQ_Value(state, action)
            Q_list.append(Q)
        if len(Q_list) == 0:
            return 0
        return max(Q_list)
    
    # update Q value for a given state action pair
    def updateQ_value(self, state, action, reward, qmax):
        q = self.getQ_Value(state,action)
        self.Q_values[(state,action)] = q + self.alpha*(reward + self.gamma*qmax - q)

    def incrementTotalEpisodes(self):
        self.episodes_total +=1

    def getEpisodesTotal(self):
        return self.episodes_total

    def getTrainingNum(self):
            return self.training_num

    def getMaxQ_Action(self, state):
        legals = state.getLegalActions()
        # in the first half of trianing, the agent is forced not to stop
        # or turn back while not being chased by the ghost
        # if self.getEpisodesTotal()*1.0/self.getTrainingNum()<0.5:
        #     if Directions.STOP in legals:
        #         legals.remove(Directions.STOP)
        #     if len(self.last_action) > 0 and state.getNumAgents() > 1:
        #         last_action = self.last_action[-1]
        #         distance0 = state.getPacmanPosition()[0]- state.getGhostPosition(1)[0]
        #         distance1 = state.getPacmanPosition()[1]- state.getGhostPosition(1)[1]
        #         if math.sqrt(distance0**2 + distance1**2) > 2:
        #             if (Directions.REVERSE[last_action] in legals) and len(legals)>1:
        #                 legals.remove(Directions.REVERSE[last_action])
            
        state_actions = Counter()
        for action in legals:
          state_actions[action] = self.getQ_Value(state, action)
        
        # Check if all Q-values are the same
        if self.all_values_same(state_actions):
             best_action =  random.choice(legals)
        else:
            # Get the action with the highest Q-value
            best_action = max(state_actions, key=state_actions.get)

        foodLoc = state.getFood()

        return best_action




    # called when the Pac Man agent is required to move
    def get_action(self, state):
        """
        Given a pacman game state, returns a legal move.

        Arguments:
        ----------
        - `state`: the current game state. See FAQ and class
                   `pacman.GameState`.

        Return:
        -------
        - A legal move as defined in `game.Directions`.
        """
        legals = state.getLegalActions()
        if (Directions.STOP in legals):
            legals.remove(Directions.STOP)

        # update Q-value for the last state action pair
        reward = state.getScore()-self.score
        if (len(self.last_state) > 0):
            last_state = self.last_state[-1]
            last_action = self.last_action[-1]
            max_q = self.getMaxQ_Values(state)
            self.updateQ_value(last_state, last_action, reward, max_q)

        # epsilon greedy
        flip = self.coinFlip(self.epsilon)
        #flip = (random.random() < self.epsilon)
        if flip:
            action =  random.choice(legals) # explore
        else:
            #action =  self.doTheRightThing(state)
            #action = random.choice(legals)  # exploit
            action = self.getMaxQ_Action(state)

        # update attributes
        self.score = state.getScore()
        self.last_state.append(state)
        self.last_action.append(action)

        #legals = state.getLegalActions()
        #legals.remove(Directions.STOP)
        #action = legals[randint(0, len(legals) - 1)]
        return action
    
    # This is called by the game after a win or a loss.
    def final(self, state):

        # update Q-values
        reward = state.getScore()-self.score
        last_state = self.lastState[-1]
        last_action = self.lastAction[-1]
        self.updateQ(last_state, last_action, reward, 0)

        # reset attributes
        self.score = 0
        self.lastState = []
        self.lastAction = []
