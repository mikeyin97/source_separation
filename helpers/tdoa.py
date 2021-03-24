import numpy as np
from .hyperboloid import *
from .signals import *
import tqdm

def build_implicit_function(pos, data, start, end, reg, sampling_rate, mic_locs_list):
    f = np.array([0., 0., 0.])
    c = 343 
    phase_diffs = []
    # r = np.random.randint(0, data.shape[0] - 1)
    for i in range(0, data.shape[0]-1):
        for j in range(i+1, data.shape[0]):
            # phase_diff = np.negative(np.abs(find_delta(data[i, start:end].T, data[j, start:end].T)))
            phase_diff = find_delta(data[i, start:end].T, data[j, start:end].T)
            phase_diffs.append(phase_diff)
            delta = (c) * (phase_diff) / sampling_rate
            f += hyperboloid_gradient(pos[0],pos[1],pos[2],mic_locs_list[i],mic_locs_list[j],delta)
    return f, phase_diffs

def solve_implicit_function(data, frames, rate, mic_locs_list, alpha = 0.01, reg = 0.01):
    ests = []
    curr_guess = np.random.rand(3)
    for f in tqdm.tqdm(range(0, data.shape[1] - frames, frames)):
        grad, phase_diffs = build_implicit_function(curr_guess, data, f, f+frames, reg, rate, mic_locs_list)
        count = 0
        while np.linalg.norm(grad + reg * curr_guess) >= 0.05 and count <= 3000:
            grad, phase_diffs = build_implicit_function(curr_guess, data, f, f+frames, reg, rate,  mic_locs_list)
            curr_guess = curr_guess - alpha * (grad + curr_guess * reg)
            count += 1
           
        print(phase_diffs)
        curr_guess = curr_guess
        ests.append(curr_guess)
    return ests

