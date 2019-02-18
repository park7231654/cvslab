# Author: Seunghyun Kim
# Date: 13 Aug 2018


import numpy as np
from scipy.ndimage import median_filter


def _get_rotmat(alpha, beta, gamma):
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    gamma = np.radians(gamma)
    sin_x, cos_x = np.sin(alpha), np.cos(alpha)
    sin_y, cos_y = np.sin(beta), np.cos(beta)
    sin_z, cos_z = np.sin(gamma), np.cos(gamma)
    rot_mat_x = np.array([
        [1, 0, 0],
        [0, cos_x, -sin_x],
        [0, sin_x, cos_x]])
    rot_mat_y = np.array([
        [cos_y, 0, sin_y],
        [0, 1, 0],
        [-sin_y, 0, cos_y]])
    rot_mat_z = np.array([
        [cos_z, -sin_z, 0],
        [sin_z, cos_z, 0],
        [0, 0, 1]])
    rotmat = np.dot(np.dot(rot_mat_z, rot_mat_y), rot_mat_x)
    return rotmat


def rotate(nparray, alpha, beta, gamma):
    if nparray.ndim != 3:
        assert False, 'The dimension of the nparray must be 3.'
    z, y, x = np.indices(nparray.shape)
    z, y, x = z.reshape(1, -1), y.reshape(1, -1), x.reshape(1, -1)
    coordinate = np.concatenate((x, y), axis=0)
    coordinate = np.concatenate((coordinate, z), axis=0)  # shape: (3, -1)
    rot_matrix = _get_rotmat(alpha, beta, gamma)
    rot_coordinate = np.rint(np.dot(rot_matrix, coordinate))  # shape: (3, -1)
    for row in rot_coordinate:
        if np.min(row) < 0:
            row += np.abs(np.min(row))
    size = np.array([
        np.max(rot_coordinate[2]),
        np.max(rot_coordinate[1]),
        np.max(rot_coordinate[0])],
        dtype=int
    )
    size += 1
    rot_nparray = np.ones(size) * np.min(nparray)
    coordinate, rot_coordinate = coordinate.T, rot_coordinate.T
    for i in range(len(coordinate)):
        p, p_ = coordinate[i], rot_coordinate[i]
        rot_nparray[int(p_[2]), int(p_[1]), int(p_[0])] = nparray[p[2], p[1], p[0]]
    rot_nparray = median_filter(rot_nparray, size=3)
    return rot_nparray


def crop(nparray, pos, crop_size, crop_margin=0):
    if nparray.ndim != 3:
        assert False, 'The dimension of the nparray must be 3.'
    if pos.shape != (3,):
        assert False, 'The shape of pos must be (3,).'
    if crop_size.shape != (3,):
        assert False, 'The shape of crop_size must be (3,).'
    half_size = np.rint(crop_size / 2)
    vmin = (pos - half_size) - crop_margin
    vmin = [np.max([0, int(ele)]) for ele in vmin]
    shape = nparray.shape
    vmax = vmin + crop_size + (crop_margin * 2)
    vmax = [np.min([ax, int(ele)]) for ax, ele in zip(shape, vmax)]
    return nparray[vmin[0]:vmax[0], vmin[1]:vmax[1], vmin[2]:vmax[2]]


def crop_center(nparray, crop_size, crop_margin=0):
    center_pos = np.rint(np.array(nparray.shape) / 2)
    center_pos = np.array([int(ele) for ele in center_pos])
    return crop(nparray, center_pos, crop_size, crop_margin)


def wrap(nparray, wrap_size):
    if nparray.ndim != 3:
        assert False, 'The dimension of the nparray must be 3.'
    shape = nparray.shape
    new_shape = wrap_size
    wrapped_nparray = np.ones(new_shape, dtype=np.int16) * np.min(nparray)
    vmin = np.rint((new_shape - shape) / 2)
    vmin = np.array([int(ele) for ele in vmin])
    vmax = vmin + shape
    wrapped_nparray[vmin[0]:vmax[0], vmin[1]:vmax[1], vmin[2]:vmax[2]] = nparray
    return wrapped_nparray
