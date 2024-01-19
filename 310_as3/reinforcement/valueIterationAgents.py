# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
    * Please read learningAgents.py before reading this.*

    A ValueIterationAgent takes a Markov decision process
    (see mdp.py) on initialization and runs value iteration
    for a given number of iterations using the supplied
    discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
        Your value iteration agent should take an mdp on
        construction, run the indicated number of iterations
        and then act according to the resulting policy.

        Some useful mdp methods you will use:
            mdp.getStates()
            mdp.getPossibleActions(state)
            mdp.getTransitionStatesAndProbs(state, action)
            mdp.getReward(state, action, nextState)
            mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for index in range(self.iterations):
            newValues = self.values.copy()
            for curState in self.mdp.getStates():
                if self.mdp.isTerminal(curState):
                    continue
                possibleActions = self.mdp.getPossibleActions(curState)
                maxVal = float("-inf")

                for action in possibleActions:
                    val = self.computeQValueFromValues(curState, action)
                    if val > maxVal:
                        maxVal = val
                newValues[curState] = maxVal
            self.values = newValues.copy()

    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
        Compute the Q-value of action in state from the
        value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        if self.mdp.isTerminal(state):
            return 0

        result = 0
        stateProbList = self.mdp.getTransitionStatesAndProbs(state, action)
        for i in range(len(stateProbList)):
            reward = self.mdp.getReward(state, action, stateProbList[i][0])
            result = (
                result
                + (reward + self.discount * self.values[stateProbList[i][0]])
                * stateProbList[i][1]
            )
        return result

    def computeActionFromValues(self, state):
        """
        The policy is the best action in the given state
        according to the values currently stored in self.values.

        You may break ties any way you see fit.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        action = None
        value = float("-inf")
        for p_action in self.mdp.getPossibleActions(state):
            if value < self.getQValue(state, p_action):
                value = self.getQValue(state, p_action)
                action = p_action
        return action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
    * Please read learningAgents.py before reading this.*

    An AsynchronousValueIterationAgent takes a Markov decision process
    (see mdp.py) on initialization and runs cyclic value iteration
    for a given number of iterations using the supplied
    discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=1000):
        """
        Your cyclic value iteration agent should take an mdp on
        construction, run the indicated number of iterations,
        and then act according to the resulting policy. Each iteration
        updates the value of only one state, which cycles through
        the states list. If the chosen state is terminal, nothing
        happens in that iteration.

        Some useful mdp methods you will use:
            mdp.getStates()
            mdp.getPossibleActions(state)
            mdp.getTransitionStatesAndProbs(state, action)
            mdp.getReward(state)
            mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        Update = self.mdp.getStates()
        for index in range(self.iterations):
            newValues = self.values
            if len(Update) == 0:
                Update = self.mdp.getStates()
            curState = Update[0]
            Update.pop(0)
            if self.mdp.isTerminal(curState):
                continue
            possibleActions = self.mdp.getPossibleActions(curState)
            maxVal = float("-inf")

            for action in possibleActions:
                val = self.computeQValueFromValues(curState, action)
                if val > maxVal:
                    maxVal = val
            newValues[curState] = maxVal
            self.values = newValues


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
    * Please read learningAgents.py before reading this.*

    A PrioritizedSweepingValueIterationAgent takes a Markov decision process
    (see mdp.py) on initialization and runs prioritized sweeping value iteration
    for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
        Your prioritized sweeping value iteration agent should take an mdp on
        construction, run the indicated number of iterations,
        and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # Extract necessary variables and settings from the environment
        mdp = self.mdp
        values = self.values
        discount = self.discount
        iterations = self.iterations
        theta = self.theta
        states = mdp.getStates()

        # Create a data structure to track predecessors for each state
        predecessors = {}
        for state in states:
            predecessors[state] = set()

        # Initialize a priority queue to manage state updates
        pq = util.PriorityQueue()

        # Perform an initial pass to compute Q-values for each state-action pair
        for state in states:
            Q_s = util.Counter()

            for action in mdp.getPossibleActions(state):
                # Compute the Q-value for the action in the current state
                transition = mdp.getTransitionStatesAndProbs(state, action)
                for nextState, prob in transition:
                    if prob != 0:
                        predecessors[nextState].add(state)

                Q_s[action] = self.computeQValueFromValues(state, action)

            # If the state is not terminal, estimate its value and update the priority queue
            if not mdp.isTerminal(state):
                maxQ_s = Q_s[Q_s.argMax()]
                diff = abs(values[state] - maxQ_s)
                pq.update(state, -diff)

        # Begin the iterative process to improve value estimates
        for i in range(iterations):
            if pq.isEmpty():
                return

            state = pq.pop()

            if not mdp.isTerminal(state):
                Q_s = util.Counter()
                for action in mdp.getPossibleActions(state):
                    Q_s[action] = self.computeQValueFromValues(state, action)

                # Update the value of the state based on the action that maximizes the Q-value
                values[state] = Q_s[Q_s.argMax()]

            # Update predecessors' values if necessary to continue the iteration
            for p in predecessors[state]:
                Q_p = util.Counter()
                for action in mdp.getPossibleActions(p):
                    Q_p[action] = self.computeQValueFromValues(p, action)

                maxQ_p = Q_p[Q_p.argMax()]
                diff = abs(values[p] - maxQ_p)

                # If the difference exceeds a threshold, update the priority queue for the predecessor
                if diff > theta:
                    pq.update(p, -diff)
