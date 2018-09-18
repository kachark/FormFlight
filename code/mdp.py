import numpy as np
import matplotlib.pyplot as plt


class TransitionFunction:

    def __init__(self, *args, **kwargs):
        pass

    def transition(self, state, action):
        """ Must be defined by child class"""
        pass
    
    
class TwoDRandomWalk(TransitionFunction):
    """
    Two dimensional random walk

    Expected Statespace is the integers^2
    Expectation actions are {(0,1), (0,-1), (1, 0), (-1, 0), (0, 0}
    which describe the directions to move on the grid

    The transition function is a bernoulli random variable with parameter p
    x -> x + a with probability p
    x -> x with probability (1 - p)
    
    """
    def __init__(self, p, *args, **kwargs):
        self.p = p
        super().__init__(args, kwargs)
        
    def transition(self, state, action):

        if np.random.rand() < self.p:
            next_state = state + action
        else:
            next_state = state

        return next_state

class StageCost:

    def __init__(self, *args, **kwargs):
        pass

class QuadraticCost(StageCost):

    def __init__(self, Q, offset, R,  *args, **kwargs):
        self.Q = Q
        self.offset = offset
        self.R = R
        super().__init__(args, kwargs)

    def evaluate(self, x, u):
        resid = x - self.offset
        out1 = np.dot(resid, np.dot(Q, resid))
        out2 = np.dot(u, np.dot(self.R, u))
        return out1 + out2

class Policy:

    def __init__(self, *args, **kwargs):
        pass

class BangBangPolicy(Policy):

    def __init__(self, target):
        self.target = target

    def evaluate(self, state):

        dists = np.abs(state - target)
        indmax = np.argmax(dists)
        action = np.zeros((self.target.shape[0]))
        action[indmax] = - np.sign(state[indmax] - target[indmax])
        return action
        
class MarkovDecisionProcess:

    def __init__(self, transitionfunction, stagecost):
        self.trans = transitionfunction
        self.stagecost = stagecost
        
    def simulate(self, x, policy, nsteps):

        result = [x]
        costs = []
        for ii in range(1, nsteps):
            action = policy.evaluate(result[-1])
            result.append(self.trans.transition(result[-1], action))
            costs.append(self.stagecost.evaluate(result[-1], action))

        return result, costs

if __name__ == "__main__":

    target = np.array([50.0, -30.0])
    Q = np.array([[1.0, 0.0], [0.0, 1.0]])
    R = np.zeros((2, 2))
    stagecost = QuadraticCost(Q, target, R)

    p = 0.5
    transitionfunction = TwoDRandomWalk(p)

    policy = BangBangPolicy(target)
    
    mdp = MarkovDecisionProcess(transitionfunction, stagecost)

    nsteps = 150
    x0 = np.array([0.0, 0.0])
    res, costs = mdp.simulate(x0, policy, nsteps)

    res = np.array(res)
    print("res.shape", res.shape)

    f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15, 5))

    ax1.plot(res[:, 0], 'o')
    ax1.set_ylabel('x')
    ax1.set_xlabel('time')

    ax2.plot(res[:, 1], 'o')
    ax2.set_ylabel('y')
    ax2.set_xlabel('time')

    ax3.plot(res[:, 0], res[:, 1], 'o')
    ax3.set_ylabel('y')
    ax3.set_xlabel('x')


    f, (ax1, ax2) = plt.subplots(1,2, figsize=(15, 5))
    ax1.plot(costs)
    ax1.set_title("Stage Cost")

    ax2.plot(np.cumsum(costs))
    ax2.set_title("Cumulative Cost")
    plt.show()
    # for ii in range()a    
    
    
