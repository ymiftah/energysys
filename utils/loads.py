import numpy as np

def simple_load(T, mean, range):
    return (mean - range/2) * np.ones(T) + range * np.random.rand(T)