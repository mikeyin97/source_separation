from scipy.signal import signaltools
import numpy as np
import matplotlib.pyplot as plt

def find_delta_with_xcorr(signal1, signal2):
    nsamples = signal1.shape[0]
    xcorr = signaltools.correlate(signal1, signal2)
    dt = np.arange(1-nsamples, nsamples)
    recovered_time_shift = dt[xcorr.argmax()]
    return xcorr, recovered_time_shift

def find_delta(signal1, signal2): # signal1 takes place after if +ve
    nsamples = signal1.shape[0]
    xcorr = signaltools.correlate(signal1, signal2)
    dt = np.arange(1-nsamples, nsamples)
    recovered_time_shift = dt[xcorr.argmax()]
    return recovered_time_shift

def get_tdoa_phase(loc1, loc2, u, freq, c): # loc1 takes place after if +ve
    return ((np.linalg.norm(u - loc1) / c) - (np.linalg.norm(u - loc2) / c))*freq

def interpolate(l, idx):
    flr = int(np.floor(idx))
    cl = int(np.ceil(idx))
    ratio = idx % 1.0
    return (1-ratio) * l[flr] + ratio * l[cl]

def get_phase_diffs(pts, mic_locs, sample_freq):
    c = 343 # metres per second
    phase_diffs = {}
    for p in pts:
        locs = list(mic_locs.values())
        phase_diff = []
        for i in range(0, len(locs)-1):
            for j in range(i + 1, len(locs)):
                tdoa_phase = get_tdoa_phase(locs[i], locs[j], p, sample_freq, c)
                phase_diff.append(tdoa_phase)
        phase_diff = np.array(phase_diff)
        phase_diffs[tuple(p)] = tuple(phase_diff)
    return phase_diffs

if __name__ == "__main__":
    pts = [np.array([0,0,1])]
    mic_locs = {1:np.array([0,0,0]), 2:np.array([0.08,0,0]), 3:np.array([-0.08,0,0]), 4:np.array([0,-0.08,0])}
    print(get_phase_diffs(pts, mic_locs, 44100))

    time = np.arange(0, 10, 0.1)
    signal2 = np.sin(time/5 * np.pi)
    signal1 = np.roll(signal2, 5)
    print(find_delta(signal1, signal2))
    plt.plot(time, signal1, c = "blue")
    plt.plot(time, signal2, c = "red")
    plt.show()