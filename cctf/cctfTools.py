import numpy as np


def getColor(shape, r, g, b):
    colors = np.zeros((shape[0], shape[1], 3), np.int)
    colors[:] = (b, g, r)
    return colors
