# CommsCharact

## 1. Description

This simple script is useful for characterizing the performance of a communication link from a SDR to a computer (usually USB or Ethernet). It essentially measures the average latency, jitter and throughput of the SDR -> computer link for different (sample rate, buffer size) combinations (SDR -> computer measurements ONLY and NOT the other way around).

<p align="center">
  <img src="https://github.com/a-r2/SDRUtils/blob/main/CommsCharact/measurements.png"%/>
</p>

The script was developed and tested using Python 3.10, having a RTLSDR dongle connected to a PC rig through one of its USB port. For this purpose, pyrtlsdr library was used. In case you are using other library to interact with your SDR, you could maintain the core of this script. However, you would need to substitute the import of pyrtlsdr library for that other library, substitute the calls for setting the sample rate and reading the buffer, and include other calls such as reading the buffer size or destroying it, if necessary.

Regarding the files, you would simply have to modify ```settings.py``` in order to select the script parameters, and then execute ```run.py```.

## 2. Requirements

* [Python 3.10 and pip](https://www.python.org/downloads/)
* [Numpy](https://github.com/numpy/numpy)
* [Matplotlib](https://github.com/matplotlib/matplotlib)
* [PyRTLSDR](https://github.com/roger-/pyrtlsdr)

## 3. Execution

Just run it! (```python3 run.py```)
