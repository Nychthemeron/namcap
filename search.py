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
    mapper = {}
    mapper[problem.getStartState()] = None

    st.push(problem.getStartState())
    while not(st.isEmpty()):
        vertex = st.pop()
        
        if (problem.isGoalState(vertex)):
            c = vertex
            l = []
            while mapper[c] != None:
                tup = mapper[c]
                l.append(tup[1])
                c = tup[0]
            l.reverse()
            print l
            return l

        else:
            neigh = problem.getSuccessors(vertex)
            # neigh.reverse()
            # neigh.sort()
            for child in neigh:
                if child[0] not in mapper:
                    st.push(child[0])
                    mapper[child[0]] = (vertex, child[1])
                    # print mapper
                
    # visited = []
    # p = dfsRecursive(problem, problem.getStartState(), st, visited, [])
    # return p
    
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
    
    for max_depth in range(0, 10000000):
        # print max_depth
        st = Stack()
        mapper = {}
        mapper[(problem.getStartState(), 0)] = None #map of (childpos, depth): (parentpos, direction, depth)
        st.push((problem.getStartState(), 0)) # stack of ((x,y) , depth)

        while not(st.isEmpty()):
            vertex = st.pop() #( (x,y) , depth )
            depth = vertex[1]

            if (problem.isGoalState(vertex[0])):
                c = vertex
                l = []
                while mapper[c] != None:
                    tup = mapper[c]
                    l.append(tup[1])
                    c = tup[0], tup[2]
                l.reverse()
                print l
                return l

            else:
                n_depth = depth + 1 # new depth
                if n_depth < max_depth:
                    neigh = problem.getSuccessors(vertex[0])
                    # neigh.reverse()
                    for child in neigh:
                        if (child[0], n_depth) not in mapper:
                            st.push((child[0], n_depth))
                            mapper[(child[0], n_depth)] = (vertex[0], child[1], depth) 


"""
    m_depth = 0
    while True:
        result = DLS(problem, problem.getStartState(), [], [], 0, m_depth)
        if result != None:
            print result
            return result
        else:
            m_depth += 1
"""

def DLS(problem, vertex, visited, path, depth, max_depth):
    # print "v: ", vertex, " m_d: ", max_depth
    cutoff = False
    if (problem.isGoalState(vertex)):
        # print path
        return path

    elif (depth == max_depth):
        return None

    if vertex not in visited:
        visited.append(vertex)

    neigh = problem.getSuccessors(vertex)
    neigh.reverse()
    depth += 1
    for child in neigh:
        if child[0] not in visited:
            newpath = []
            for p in path:
                newpath.append(p)
            newpath.append(child[1])

            p = DLS(problem, child[0], visited, newpath, depth, max_depth)
            if p != None:
                return p


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
        succs = problem.getSuccessors(point)
        succs.reverse()
        for child in succs:
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
    from util import PriorityQueue
    pq = PriorityQueue()
    # visited = []
    mapper = {}
    costs = {}
    start = problem.getStartState()
    mapper[start] = None
    costs[start] = 0
    pq.push(start, 0)

    while not (pq.isEmpty()):
        # print costs
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
        for child in problem.getSuccessors(point):
            if not child[0] in mapper:
                cost = costs[point] + child[2]
                if (child not in costs) or (cost < costs[child[0]]):
                    costs[child[0]] = cost
                    full_cost = cost + heuristic(child[0], problem)
                    pq.push(child[0], full_cost)
                    mapper[child[0]] = point, child[1]


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
ids = iterativeDeepeningSearch
