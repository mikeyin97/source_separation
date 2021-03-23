import math
import numpy as np 
import matplotlib.pyplot as plt
import tqdm
from .signals import *

def fibonacci_sphere(samples=1):
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        points.append((x, y, z))
    return points

def lookup(data, phase_diffs, frames):
    curr_locs = []
    for f in tqdm.tqdm(range(0, data.shape[1] - frames, frames//4)):
        frame_data = data[:, f:f+(frames)]
        phase_diff = []
        x_corrs = []
        count = 0
        for i in range(0, frame_data.shape[0] - 1):
            for j in range(i+1, frame_data.shape[0]):
                x_1 = np.squeeze(np.array(frame_data[i]).T)
                x_2 = np.squeeze(np.array(frame_data[j]).T)
                xcorr, expected_phase = find_delta_with_xcorr(x_1, x_2)
                x_corrs.append(xcorr)
        x_corrs = np.array(x_corrs)
        # compare to sphere
        mid = x_corrs.shape[1] // 2
        curr_loc = None
        max_sum = 0
        for loc in phase_diffs:
            curr_sum = 0
            indices = phase_diffs[loc]
            for j in range(len(indices)):
                index = mid + indices[j]
                val = interpolate(x_corrs[j], index)
                curr_sum += val
            if curr_sum > max_sum:
                max_sum = curr_sum
                curr_loc = loc         
        curr_locs.append(curr_loc)
    return curr_locs

if __name__ == "__main__":
    pts = fibonacci_sphere(360)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in pts:
        ax.scatter(i[0], i[1], i[2])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()