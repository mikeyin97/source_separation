import numpy as np

def grad(W, x):
    x_h = x.H
    R_xx = np.matmul(x, x_h)
    R_xx = R_xx

    R_yy = np.matmul(np.matmul(W, R_xx), W.H)
    
    # if W singular, perturb slightly until nonsingular
    A = np.asmatrix(np.linalg.lstsq(W, np.identity(2))[0])

    E = R_yy - np.diag(np.diag(R_yy))

    dJ1 = 4*np.matmul(np.matmul(E, W), R_xx)
    dJ2 = 2*np.matmul((np.matmul(W, A) - np.identity(W.shape[0])), A.H)

    
    alpha = (np.linalg.norm(R_xx))**-2
    grad = alpha * dJ1 + dJ2
    
    return grad

def grad_descent(W_init, x, mu, reg = 0.01):
    W = W_init
    count = 0
    g = grad(W, x)
    while np.linalg.norm(g) >= 0.001:
        g = grad(W, x)
        W = (1-mu*reg) * W - mu*g 
        count += 1
    return W