import numpy as np 

def hyperboloid(x,y,z,mic1,mic2,delta):
    return np.sqrt(np.square(mic1[0] - x) + np.square(mic1[1] - y) + 
        np.square(mic1[2] - z)) - np.sqrt(np.square(mic2[0] - x) + 
        np.square(mic2[1] - y) + np.square(mic2[2] - z)) - delta

def hyperboloid_gradient(x,y,z,mic1,mic2,delta):
    denom1 = np.sqrt(np.square(mic1[0] - x) + np.square(mic1[1] - y) + np.square(mic1[2] - z))
    denom2 = np.sqrt(np.square(mic2[0] - x) + np.square(mic2[1] - y) + np.square(mic2[2] - z))
    x_grad = 2 * hyperboloid(x,y,z,mic1,mic2,delta) * ((x-mic1[0]) / denom1  -  (x-mic2[0]) / denom2)
    y_grad = 2 * hyperboloid(x,y,z,mic1,mic2,delta) * ((y-mic1[1]) / denom1  -  (y-mic2[1]) / denom2)
    z_grad = 2 * hyperboloid(x,y,z,mic1,mic2,delta) * ((z-mic1[2]) / denom1  -  (z-mic2[2]) / denom2)
    return np.array([x_grad, y_grad, z_grad])

def hyperbola(x,y,mic1,mic2,delta):
    return np.sqrt(np.square(mic1[0] - x) + np.square(mic1[1] - y)) - np.sqrt(np.square(mic2[0] - x) + np.square(mic2[1] - y)) - delta

def hyperbola_gradient(x,y,mic1,mic2,delta):
    x_grad = 2 * hyperbola(x,y,mic1,mic2,delta) * ((x-mic1[0]) / np.sqrt(np.square(mic1[0] - x) + np.square(mic1[1] - y))  -  (x-mic2[0]) / np.sqrt(np.square(mic2[0] - x) + np.square(mic2[1] - y)))
    y_grad = 2 * hyperbola(x,y,mic1,mic2,delta) * ((y-mic1[1]) / np.sqrt(np.square(mic1[0] - x) + np.square(mic1[1] - y))  -  (y-mic2[1]) / np.sqrt(np.square(mic2[0] - x) + np.square(mic2[1] - y)))
    return np.array([x_grad, y_grad])

# print(hyperbola_gradient(1, -1, [1,2], [1, -2], 2))