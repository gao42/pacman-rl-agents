# Complete this class for all parts of the project
from pacman_module.game import Actions
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

        self.alpha = float(0.2)    # the learning rate
        self.epsilon = float(0.05) # the exploration rate
        self.gamma = float(0.8)    # the discount factor

        # Q values
        #self.Q_values = dict() # or util.Counter()
        self.Q_values = Counter()

        # feature weights
        self.weights = Counter()

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
    
    def allValuesSame(self, counter):
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
    
    def nearestFoodDist(self, state, posit, food, walls):
        grid = [(posit[0], posit[1], 0)]
        neighbors = set()
        while grid:
            posit_x, posit_y, dist = grid.pop(0)
            if (posit_x, posit_y) not in neighbors:
                neighbors.add((posit_x, posit_y))

                if food[posit_x][posit_y]:
                    return dist

                avaiable_neighbors = Actions.getLegalNeighbors((posit_x, posit_y), walls)
                for nbr_x, nbr_y in avaiable_neighbors:
                    grid.append((nbr_x, nbr_y, dist+1))
        return None
    
    def nearestGhostDist(self, state, posit, ghost, walls):
        # grid = [(posit[0], posit[1], 0)]
        # neighbors = set()
        # while grid:
        #     posit_x, posit_y, dist = grid.pop(0)
        #     if (posit_x, posit_y) not in neighbors:
        #         neighbors.add((posit_x, posit_y))

        #         if ghost == [(posit_x, posit_y)]:
        #             return dist

        #         avaiable_neighbors = Actions.getLegalNeighbors((posit_x, posit_y), walls)
        #         for nbr_x, nbr_y in avaiable_neighbors:
        #             grid.append((nbr_x, nbr_y, dist+1))
        # return None

        return math.sqrt((posit[0]-ghost[0])**2 + (posit[1]-ghost[1])**2)

    def getFeatures(self, state, action):
        """
        Returns a Counter of features for a given state and action.
        
        Args:
        state (GameState): The current game state.
        action (str): The action to be taken.
        
        Returns:
        Counter: A Counter object with features as keys and feature values as values.
        """
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()
        #capsule = state.getCapsules()

        pacman_x, pacman_y = state.getPacmanPosition()
        delta_x, delta_y = Actions.directionToVector(action)
        next_x, next_y = int(pacman_x + delta_x), int(pacman_y + delta_y)

        neighbors = Actions.getLegalNeighbors(state.getPacmanPosition(), walls)

        features = Counter()

        features["bias"] = 1.0
        features["nearest-food"] = 0.0
        features["nearest-ghost"] = 0.0 
        features["num-ghosts-1-move-away"] = 0.0
        features["eats-food"] = 0.0

        nearest_food_dist = self.nearestFoodDist(state, neighbors[0], food, walls)
        features["nearest-food"] = nearest_food_dist
        
        nearest_ghost_dist = self.nearestGhostDist(state, neighbors[0], ghosts[0], walls)
        features["nearest-ghost"] = nearest_ghost_dist

        num_ghosts_1_move_away = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)
        features["num-ghosts-1-move-away"] = num_ghosts_1_move_away

        if not features["num-ghosts-1-move-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        for f in features:
            features[f] /= 10.0 # scale down

        return features

    # returns the Q value for a given state
    def getQ_Value(self, state, action):
        return self.Q_values[(state,action)]
    
    # returns the Q value for a given state based on the feature weights
    def getFeaturesQ_Value(self, state, action):
        features = self.getFeatures(state,action)
        Q_Value = 0.0

        for feature in features:
            Q_Value += self.weights[feature] * features[feature]

        return Q_Value
    
    # return the maximum Q values of each state
    def getMaxQ_Values(self, state):
        q_list = []
        #for action in state.getLegalPacmanActions():
        for action in state.getLegalActions():
            q_value = self.getQ_Value(state, action)
            q_list.append(q_value)
        if len(q_list) == 0:
            return 0
        return max(q_list)
    
        # return the maximum Q values of each state
    def getMaxFeaturesQ_Values(self, state):
        q_list = []
        #for action in state.getLegalPacmanActions():
        for action in state.getLegalActions():
            #Q = self.getQ_Value(state, action)
            q_value = self.getFeaturesQ_Value(state, action)
            q_list.append(q_value)
        if len(q_list) == 0:
            return 0
        return max(q_list)
    
    # update Q value for a given state action pair
    def updateQ_Value(self, state, action, reward, q_max):
        q_value = self.getQ_Value(state,action)
        self.Q_values[(state,action)] = q + self.alpha*(reward + self.gamma*q_max - q)

        # update Q value for a given state action pair
    def updateFeaturesQ_Value(self, state, action, reward, q_max):
        q_value = self.getFeaturesQ_Value(state,action)
        #self.Q_values[(state,action)] = q + self.alpha*(reward + self.gamma*qmax - q)

        #difference = reward + (self.discount * self.computeValueFromQValues(nextState) - self.getQValue(state, action))
        difference = reward + self.gamma*q_max - q_value
        features = self.getFeatures(state, action)

        for feature in features:
            self.weights[feature] += self.alpha * features[feature] * difference

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
        if self.allValuesSame(state_actions):
             best_action =  random.choice(legals)
        else:
            # Get the action with the highest Q-value
            best_action = max(state_actions, key=state_actions.get)

        return best_action
    
    def getMaxFeaturesQ_Action(self, state):
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
          state_actions[action] = self.getFeaturesQ_Value(state, action)
        
        # Check if all Q-values are the same
        if self.allValuesSame(state_actions):
             best_action =  random.choice(legals)
        else:
            # Get the action with the highest Q-value
            best_action = max(state_actions, key=state_actions.get)

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
            #max_q = self.getMaxQ_Values(state)
            #self.updateQ_Value(last_state, last_action, reward, max_q)
            max_q = self.getMaxFeaturesQ_Values(state)
            self.updateFeaturesQ_Value(last_state, last_action, reward, max_q)

        # epsilon greedy
        flip = self.coinFlip(self.epsilon)
        #flip = (random.random() < self.epsilon)
        if flip:
            action =  random.choice(legals) # explore
        else:
            #action =  self.doTheRightThing(state)
            #action = random.choice(legals)  # exploit
            action = self.getMaxFeaturesQ_Action(state)

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
