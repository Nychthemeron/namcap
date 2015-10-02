# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    from util import Stack
    
    st = Stack()
    visited = []
    p = dfsRecursive(problem, problem.getStartState(), st, visited, [])
    return p
    
    # pathfind = {}
    # st.push(problem.getStartState())
    # iterative approach:
    # while (not st.isEmpty()):
    #     point = st.pop() # (x,y)
    #     if problem.isGoalState(point):
    #         # print point
    #         print pathfind
    #         # print visited
    #     elif (not (point in visited)):
    #         visited.append(point)
    #         # print pathfind, '\n'
    #         print visited, '\n'
    #         for child in problem.getSuccessors(point):
    #             st.push(child[0])
    #             pathfind[child[0]] = point #this preemptively adds!
    # util.raiseNotDefined()

def dfsRecursive(problem, vertex, stack, visited, path):
    if (problem.isGoalState(vertex)):
        print path
        return path

    elif (not (vertex in visited)):
        visited.append(vertex)
        next_v = problem.getSuccessors(vertex)
        next_v.reverse() # to keep ordering
        if len(next_v) > 0:
            for child in next_v:
                stack.push(child[0])
                #copy path to avoid reassignment
                newpath = []
                for p in path:
                    newpath.append(p)
                newpath.append(child[1])
                new_v = stack.pop()
                #get our path if we hit
                p = dfsRecursive(problem, new_v, stack, visited, newpath)
                if p: return p

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue
    q = Queue()
    mapper = {} #child_point : (parent_point, direction_to_child)
    q.push(problem.getStartState())
    mapper[problem.getStartState()] = None #root

    while (not q.isEmpty()):
        point = q.pop()

        if (problem.isGoalState(point)):
            c = point
            l = []
            while mapper[c] != None:
                tup = mapper[c]
                l.append(tup[1])
                c = tup[0]
            l.reverse()
            print l
            return l

        else:
            for child in problem.getSuccessors(point):
                if (child[0] not in mapper):
                    q.push(child[0])
                    mapper[child[0]] = (point, child[1])

    # util.raiseNotDefined()

def iterativeDeepeningSearch(problem):
    """Iterative Deepening version of DFS"""
    from util import Stack

    st = Stack()
    visited = []
    mapper = {}
    # max_depth = 100
    depth = 0
    while True:
        result = DLS(problem, problem.getStartState(), st, visited, mapper, 0, depth)
        if result != None:
            print result
            return result
        else:
            #clear it and try again
            st = Stack()
            visited = []
            mapper = {}
            depth += 1

    # util.raiseNotDefined()

def DLS(problem, vertex, stack, visited, path, depth, max_depth):
    # print "v: ", vertex, " m_d: ", max_depth
    if (problem.isGoalState(vertex)):
        # print path
        return path

    elif (depth is max_depth):
        return None

    elif (not (vertex in visited)):
        visited.append(vertex)
        new_depth = depth + 1
        next_v = problem.getSuccessors(vertex)
        next_v.reverse() # to keep ordering
        if len(next_v) > 0:
            for child in next_v:
                stack.push(child[0])
                #copy path to avoid reassignment
                newpath = []
                for p in path:
                    newpath.append(p)
                newpath.append(child[1])
                new_v = stack.pop()
                #get our path if we hit
                p = DLS(problem, new_v, stack, visited, newpath, new_depth, max_depth)
                if p: return p

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue

    pq = PriorityQueue()
    visited = []
    start = problem.getStartState()
    mapper = {}
    
    mapper[problem.getStartState()] = None
    pq.push(problem.getStartState(), 1)

    while (not pq.isEmpty()):
        point = pq.pop()
        if problem.isGoalState(point):
            current = point
            l = []
            while mapper[current] != None:
                tup = mapper[current]
                l.append(tup[1])
                current = tup[0]
            l.reverse()
            print l
            return l
            #util.raiseNotDefined()
        if not (point in visited):
            visited.append(point)
        for child in problem.getSuccessors(point):
            if not (child[0] in mapper):
                pq.push(child[0], child[2]) #child has (xy, direction, weight)
                mapper[child[0]] = point, child[1]
    # util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
ids = iterativeDeepeningSearch
