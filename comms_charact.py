import matplotlib.pyplot as plt
import numpy as np
import time

from rtlsdr import RtlSdr

sdr = RtlSdr() #RTLSDR object

# PARAMETERS
ITER_NUM     = 10 #number of iterations to calculate time performance for a given (sample_rate, buffer_size) combination
RESULTS_FILE = "measurements" #.npy file that stores (latency, jitter, throughput) measurements
sample_rates = np.array([240e3, 960e3, 1.2e6, 1.6e6, 1.92e6, 2.44e6, 2.88e6, 3.2e6], dtype=int) #sample rates to sweep
buffer_sizes = np.array([256 * 2 ** exp for exp in range(14)], dtype=int) #buffer sizes to sweep

# LOOP
# Print sample rates and buffer sizes to sweep
print("Sample rates: " + str(sample_rates))
print("Buffer sizes: " + str(buffer_sizes))
# Define constants
SR_LEN = len(sample_rates)
BS_LEN = len(buffer_sizes)
# Initialize arrays
latency    = np.NaN * np.ones((SR_LEN, BS_LEN)) #average latency
jitter     = np.NaN * np.ones((SR_LEN, BS_LEN)) #average jitter
throughput = np.NaN * np.ones((SR_LEN, BS_LEN)) #average throughput
# Get measurements from sample rates - buffer sizes combinations
for ind_sr, cur_sr in enumerate(sample_rates):
    sdr.set_sample_rate(cur_sr)
    # Check if current sample rate is indeed set in the SDR
    if np.mod(sdr.get_sample_rate(), cur_sr) < 1:
        print("Testing sample rate: " + str(np.around(int(sdr.get_sample_rate()) * 1e-6, decimals=6)) + " MHz...")
    while not np.mod(sdr.get_sample_rate(), cur_sr) < 1:
        sdr.set_sample_rate(cur_sr)
    for ind_bs, cur_bs in enumerate(buffer_sizes):
        if cur_bs > cur_sr: #ignore when buffer size is greater than sample rate
            break
        print("Testing buffer size: " + str(cur_bs) + " samples...") #there is no way to check that current buffer size is set in the SDR
        diff_time = np.NaN * np.ones(ITER_NUM) #initialize array
        for i in range(ITER_NUM):
            start_time   = time.perf_counter()
            iq           = sdr.read_samples(cur_bs) #get buffer size samples
            end_time     = time.perf_counter()
            diff_time[i] = end_time - start_time
        latency[ind_sr, ind_bs]    = np.mean(diff_time) #mean
        jitter[ind_sr, ind_bs]     = np.std(diff_time) #standard deviation
        throughput[ind_sr, ind_bs] = cur_bs / latency[ind_sr, ind_bs]
# Transform units
latency    *= 1e3 #t -> ms
jitter     *= 1e3 #t -> ms
throughput *= 1e-6 #Hz -> MHz

# SAVE
np.save(RESULTS_FILE, (latency, jitter, throughput))

# PLOTS
cm                   = plt.get_cmap("hsv") #color map
fig, (ax1, ax2, ax3) = plt.subplots(3, num="SDR COMM characterization") #figure
# Plot 1
ax1.set_prop_cycle("color", [cm(1.*i/BS_LEN) for i in range(BS_LEN)])
ax1.plot(sample_rates, latency, marker="o")
box1 = ax1.get_position()
ax1.set_position([box1.x0, box1.y0, box1.width * 0.9, box1.height])
ax1.set_ylabel("Latency [ms]")
ax1.set_xlabel("Sample rate [MHz]")
ax1.set_xticks(sample_rates, labels=np.around(sample_rates * 1e-6, decimals=6))
ax1.grid()
# Plot 2
ax2.set_prop_cycle("color", [cm(1.*i/BS_LEN) for i in range(BS_LEN)])
ax2.plot(sample_rates, jitter, marker="o")
box2 = ax2.get_position()
ax2.set_position([box2.x0, box2.y0, box2.width * 0.9, box2.height])
ax2.set_ylabel("Jitter [ms]")
ax2.set_xlabel("Sample rate [MHz]")
ax2.set_xticks(sample_rates, labels=np.around(sample_rates * 1e-6, decimals=6))
ax2.grid()
# Plot 3
ax3.set_prop_cycle("color", [cm(1.*i/BS_LEN) for i in range(BS_LEN)])
ax3.plot(sample_rates, throughput, marker="^")
ax3.plot(sample_rates, sample_rates * 1e-6, marker=".", color="black") #sample rate required for real-time processing
box3 = ax3.get_position()
ax3.set_position([box3.x0, box3.y0, box3.width * 0.9, box3.height])
ax3.set_ylabel("Throughput [MS/s]")
ax3.set_xlabel("Sample rate [MHz]")
ax3.set_xticks(sample_rates, labels=np.around(sample_rates * 1e-6, decimals=6))
ax3.grid()
# Legend
legend_list = np.append([str(bs) for bs in buffer_sizes], "Required")
ax3.legend(legend_list, title="Buffer size [S]", loc="center left", bbox_to_anchor=(1, 1.7)) #only one legend for all subplots
# Show figure
plt.show()
