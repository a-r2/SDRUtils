import numpy as np

# PARAMETERS
ITER_NUM     = 10 #number of iterations to calculate time performance for a given (sample_rate, buffer_size) combination
RESULTS_FILE = "measurements" #.npy file that stores (latency, jitter, throughput) measurements
sample_rates = np.array([240e3, 960e3, 1.2e6, 1.6e6, 1.92e6, 2.44e6, 2.88e6, 3.2e6], dtype=int) #sample rates to sweep
buffer_sizes = np.array([256 * 2 ** exp for exp in range(14)], dtype=int) #buffer sizes to sweep