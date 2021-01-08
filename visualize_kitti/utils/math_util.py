import numpy as np

def rotx(t):
    """
    3D Rotation about the x-axis.
    :param t:
    :return:
    """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])


def roty(t):
    """
    http://planning.cs.uiuc.edu/node102.html
    Rotation about the y-axis.
    :param t:
    :return: matrix
    """
    '''  '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def rotz(t):
    """
    Rotation about the z-axis.
    :param t:
    :return:
    """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def rotation_translation_matrix(rx, ry, rz, t):
    """

    :param rx: in kitti: pitch angle in radian and rx=0
    :param ry: in kitti: yaw angle in radian
    :param rz: in kitti: roll angle in radian and rz=0
    :param t: (3,) translation from object to rect cam coord
    :return: matrix (3x4)
    """
    R = np.dot(rotz(rz), np.dot(roty(ry), rotx(rx)))
    t = np.array(t).reshape([len(t), 1])
    # rotate and translate 3d bounding box [R|t] (3, 4)
    Rt_matrix = np.hstack((R, t))
    return Rt_matrix