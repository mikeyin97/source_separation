import numpy as np

def build_implicit_function(pos, data, start, end, reg):
    f = np.array([0., 0., 0.])
    c = 343 
    for i in range(0, data.shape[0] - 1):
        for j in range(i+1, data.shape[0]):
            # if positive phase_diff, first comes after second
            phase_diff = find_delta(data[i, start:end].T, data[j, start:end].T)
            delta = - (c) * (phase_diff) / sampling_rate
            f += hyperboloid_gradient(pos[0],pos[1],pos[2],mic_locs_list[i],mic_locs_list[j],delta)
    return f

def solve_implicit_function(data, frames, alpha = 0.05, reg = 0.01):
    ests = []
    curr_guess = np.array([0,1,0])
    for f in tqdm(range(0, data.shape[1] - frames, frames)):
        grad = build_implicit_function(curr_guess, data, f, f+frames, reg)
        count = 0
        while np.linalg.norm(grad + reg * curr_guess) >= 0.04 and count <= 10000:
            print(np.linalg.norm(grad + reg * curr_guess))
            grad = build_implicit_function(curr_guess, data, f, f+frames, reg)
            curr_guess = curr_guess - alpha * (grad + curr_guess * reg)
            count += 1
        print(count)
        print(curr_guess)
        curr_guess = curr_guess
        ests.append(curr_guess)
        # normalize to depth 1
    return ests