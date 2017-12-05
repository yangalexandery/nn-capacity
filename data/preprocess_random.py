import numpy as np

sizes = [100, 500, 1000, 5000, 10000, 60000]
dimension = 32 * 32 * 3
num_labels = 10

for size in sizes:
    data = np.random.randn(size, dimension)
    labels = np.random.randint(0, num_labels, (size, 1))
    np.savez('random_{}.npz'.format(size), data, labels)
