import numpy as np
import math


#https://github.com/fhirschmann/rdp/blob/master/rdp/__init__.py



"""
    pldist is a function directly copied from link above, to calculate the perpendicular distance.
"""
def pldist(point, start, end):
    """
    Calculates the distance from ``point`` to the line given
    by the points ``start`` and ``end``.
    :param point: a point
    :type point: numpy array
    :param start: a point of the line
    :type start: numpy array
    :param end: another point of the line
    :type end: numpy array
    """
    if np.all(np.equal(start, end)):
        return np.linalg.norm(point - start)

    return np.divide(
            np.abs(np.linalg.norm(np.cross(end - start, start - point))),
            np.linalg.norm(end - start))


"""
    rdp_fixed_num is a rewritten version of rdp taken from github link in top of the file.
    This version does not depend on a epsilon, however removes the points least affecting the 
    strokes' structure, until a fixed limit is reached. Its purpose is to normalize all traces
    to a fixed length.
"""
def rdp_fixed_num(M, fixed_num, dist=pldist):
    min_distance = math.inf
    index = -1
    indices = np.ones(len(M), dtype=bool)

    for i in range(0, len(M) - 2):
        d = dist(M[i + 1], M[i], M[i + 2])
        if d < min_distance:
            index = i + 1
            min_distance = d

    indices[index] = False

    if len(M) - 1 <= fixed_num:
        return M[indices]
    else:
        return rdp_fixed_num(M[indices], fixed_num)

    

    

