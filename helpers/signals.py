from scipy.signal import signaltools

def find_delta_with_xcorr(signal1, signal2):
    nsamples = signal1.shape[0]
    xcorr = signaltools.correlate(signal1, signal2)
    dt = np.arange(1-nsamples, nsamples)
    recovered_time_shift = dt[xcorr.argmax()]
    return xcorr, recovered_time_shift

def find_delta(signal1, signal2):
    nsamples = signal1.shape[0]
    xcorr = signaltools.correlate(signal1, signal2)
    dt = np.arange(1-nsamples, nsamples)
    recovered_time_shift = dt[xcorr.argmax()]
    return recovered_time_shift

def get_tdoa_phase(loc1, loc2, u, freq, c):
    return freq/c*np.dot((loc1 - loc2), u)

def interpolate(l, idx):
    flr = int(np.floor(idx))
    cl = int(np.ceil(idx))
    ratio = idx % 1.0
    return (1-ratio) * l[flr] + ratio * l[cl]