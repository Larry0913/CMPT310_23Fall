# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        distMin = float("inf")
        currentFood = currentGameState.getFood()
        newFood = currentFood.asList()
        for food in newFood:
            cur_dist = manhattanDistance(food, newPos)
            for ghostPosition in successorGameState.getGhostPositions():
                if newPos == ghostPosition:
                    return -float("inf")
            if cur_dist < distMin:
                distMin = cur_dist
        if action == "Stop":
            distMin = float("inf")
        return -distMin


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        largest_score = float("-inf")
        direction = None
        actions = gameState.getLegalActions(0)

        for action in actions:
            current_successor = gameState.generateSuccessor(0, action)
            score = self.miniMaxSearch(current_successor, 1)

            if score > largest_score:
                largest_score = score
                direction = action

        return direction

    def miniMaxSearch(self, gameState, depth):
        agentsNum = gameState.getNumAgents()
        actions = gameState.getLegalActions(depth % agentsNum)

        if depth == self.depth * agentsNum or gameState.isLose() or gameState.isWin():
            score = self.evaluationFunction(gameState)
            return score

        if depth % agentsNum != 0:
            score = float("inf")
            for action in actions:
                current_successor = gameState.generateSuccessor(
                    depth % agentsNum, action
                )
                curr_score = self.miniMaxSearch(current_successor, depth + 1)
                if curr_score < score:
                    score = curr_score
            return score

        else:
            score = -float("inf")
            for action in actions:
                current_successor = gameState.generateSuccessor(
                    depth % agentsNum, action
                )
                curr_score = self.miniMaxSearch(current_successor, depth + 1)
                if curr_score > score:
                    score = curr_score
            return score


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        alpha = -float("inf")
        beta = float("inf")
        direction = None
        actions = gameState.getLegalActions(0)

        for action in actions:
            current_successor = gameState.generateSuccessor(0, action)
            score = self.alphaBeta(current_successor, 1, alpha, beta)
            if score > alpha:
                alpha = score
                direction = action
        return direction

    def alphaBeta(self, gameState, depth, alpha, beta):
        agentsNum = gameState.getNumAgents()
        actions = gameState.getLegalActions(depth % agentsNum)

        if depth == self.depth * agentsNum or gameState.isLose() or gameState.isWin():
            score = self.evaluationFunction(gameState)
            return score

        if depth % agentsNum != 0:
            score = float("inf")
            for action in actions:
                current_successor = gameState.generateSuccessor(
                    depth % agentsNum, action
                )
                curr_score = self.alphaBeta(current_successor, depth + 1, alpha, beta)
                if curr_score < score:
                    score = curr_score
                beta = min(beta, score)
                if alpha > beta:
                    break
            return score

        else:
            score = -float("inf")
            for action in actions:
                current_successor = gameState.generateSuccessor(
                    depth % agentsNum, action
                )
                curr_score = self.alphaBeta(current_successor, depth + 1, alpha, beta)
                if curr_score > score:
                    score = curr_score
                alpha = max(alpha, score)
                if alpha > beta:
                    break
            return score


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        score, action = self.expectiMax(gameState, self.depth, 0)
        return action

    def expectiMax(self, gameState, depth, agent_index):
        best_action = None
        best_score = -float("inf")
        total_score = 0
        if agent_index == gameState.getNumAgents():
            agent_index = 0
            depth -= 1

        actions = gameState.getLegalActions(agent_index)

        if depth == 0 or not actions:
            return self.evaluationFunction(gameState), None

        for action in actions:
            successor_state = gameState.generateSuccessor(agent_index, action)
            if agent_index == 0:
                score, _ = self.expectiMax(successor_state, depth, agent_index + 1)
                if score > best_score:
                    best_score = score
                    best_action = action
            else:
                successor_state = gameState.generateSuccessor(agent_index, action)
                score, _ = self.expectiMax(successor_state, depth, agent_index + 1)
                total_score += score

        if agent_index == 0:
            return best_score, best_action
        else:
            return total_score / len(actions), None


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: We aim to revise the evaluation criteria, focusing on Pacman's proximity to the ghosts.
    Considering the presence of multiple ghosts, we no longer prioritize just one ghost's distance but rather all the ghosts.
    Therefore, for each potential move, we need to assign a substantial penalty to the score if that move results in Pacman encountering a ghost.
    In simpler terms, if Pacman manages to avoid ghosts, the impact on the score is relatively minor,
    but if they come too close to a ghost, a significant penalty is imposed to discourage such encounters.
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    pacman_position = currentGameState.getPacmanPosition()
    ghost_states = currentGameState.getGhostStates()
    curScaredTime = [ghostState.scaredTimer for ghostState in ghost_states]
    currentFood = currentGameState.getFood()
    foodList = currentFood.asList()
    foodDistance = sum([manhattanDistance(food, pacman_position) for food in foodList])
    ghostDist = sum(
        [
            manhattanDistance(ghost.getPosition(), pacman_position)
            for ghost in ghost_states
        ]
    )
    if ghostDist == 0:
        return -float("inf")
    score = currentGameState.getScore()
    if foodDistance > 0:
        score += 1.0 / foodDistance
    score += len(foodList)
    score += sum(curScaredTime)

    return score


# Abbreviation
better = betterEvaluationFunction
