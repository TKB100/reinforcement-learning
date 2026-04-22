import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sps

arms = [(1, 2), (3, 4), (5, 6)]
mySamples = []
plt.figure(figsize=(12, 8))

for i, (shape, scale) in enumerate(arms):
    samples = np.random.gamma(shape, scale, 1000)
    mySamples.append(samples)

    plt.subplot(1, 3, i + 1)

    count, bins, ignored = plt.hist(samples, 50, density=True, alpha=0.6)
    y = bins ** (shape - 1) * (np.exp(-bins / scale) / (sps.gamma(shape) * scale ** shape))
    plt.plot(bins, y, linewidth=2, color='r', label='Theoretical PDF')
    plt.legend()
    plt.title(f"Arm {i+1}: Shape = {shape}, Scale = {scale}")

plt.show()
