import numpy as np
import matplotlib.pyplot as plt

arms = [(1, 2), (3, 4), (5, 6)]
nArms = 3
nGames = 1000
optimalArm = 2

#ε-greedy arm
def experiment(epsilon):
    Q = np.zeros(nArms)
    N = np.zeros(nArms)
    reward = []
    optimal = []

    for game in range(nGames):
        if np.random.rand() < epsilon:
            action = np.random.randint(nArms)
        else:
            action = np.argmax(Q)

        shape, scale = arms[action]
        r = np.random.gamma(shape, scale)
        reward.append(r)
        optimal.append(action == optimalArm)

        N[action] += 1
        Q[action] += (r - Q[action]) / N[action]

        if game >= 200:
            if np.mean(reward[-100:]) == np.mean(reward[-200:-100]):
                break

    return reward, optimal

#ucb arm
def ucb():
    Q = np.zeros(nArms)
    N = np.zeros(nArms)
    reward = []
    optimal = []

    for game in range(nGames):
        ucbVals = Q + 2 * np.sqrt(np.log(game + 1) / (N + 1e-5)) #Given formula 
        action = np.argmax(ucbVals)

        shape, scale = arms[action]
        r = np.random.gamma(shape, scale)
        reward.append(r)
        optimal.append(action == optimalArm)

        N[action] += 1
        Q[action] += (r - Q[action]) / N[action]

        if game >= 200:
            if np.mean(reward[-100:]) == np.mean(reward[-200:-100]):
                break

    return reward, optimal

#average 
def average(data):
    return np.cumsum(data) / np.arange(1, len(data) + 1)



r1, o1 = experiment(epsilon=0.01)
r2, o2 = experiment(epsilon=0.1)
r3, o3 = experiment(epsilon=0.5)
rucb, oucb = ucb()


#plotting
plt.figure(figsize=(12, 5))
plt.plot(average(o1), label='epsilon=0.01')
plt.plot(average(o2), label='epsilon=0.1')
plt.plot(average(o3), label='epsilon=0.5')
plt.plot(average(oucb), label='UCB')

plt.xlabel('Game')
plt.ylabel('Optimal Action')
plt.title('Optimal Action Over Time')
plt.legend()
plt.savefig('task2_action.png', dpi=150, bbox_inches='tight')
plt.show()
