import scipy.ndimage

def laplace_var(x):
    return scipy.ndimage.laplace(x).var()
