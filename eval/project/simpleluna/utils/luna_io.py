# Author: Seunghyun Kim
# Date: 24 Aug 2018


import os
import shutil

import numpy as np
import pandas as pd
from scipy.ndimage import interpolation

from . import file_io


def get_subsets(data_dir):
    subsets = file_io.get_dirs(data_dir, regex='subset*')
    return subsets


def get_files_from_subsets(subsets, regex='*'):
    files = []
    for subset in subsets:
        subset_files = file_io.get_files(subset, regex)
        for file in subset_files:
            files.append(file)
    return files


def get_dataframe(csv_file):
    df = pd.read_csv(csv_file)
    return df


def get_mhd_dataframe(mhds, csv_file):
    def _helper(case):
        nonlocal mhds
        for name in mhds:
            if case in name:
                return name
    df = pd.read_csv(csv_file)
    df['file'] = df['seriesuid'].apply(_helper)
    df = df.dropna()
    return df


def _real_resize_factor(nparray_shape, spacing, adj_spacing=None):
    if adj_spacing is None:
        adj_spacing = list(np.ones_like(nparray_shape))
    new_nparray_shape = np.round(nparray_shape * (spacing / adj_spacing))
    real_resize_factor = new_nparray_shape / nparray_shape
    return real_resize_factor


def resample(nparray, spacing, only_spacing=False):
    real_resize_factor = _real_resize_factor(nparray.shape, spacing)
    new_spacing = spacing / real_resize_factor
    resampled_nparray = None
    if only_spacing:
        pass
    else:
        resampled_nparray = interpolation.zoom(nparray, real_resize_factor, mode='nearest')
    return resampled_nparray, new_spacing


def resize_diameter(nparray_shape, spacing, diameter):
    real_resize_factor = _real_resize_factor(nparray_shape, spacing)
    new_spacing = spacing / real_resize_factor
    diam = int(np.rint(diameter / new_spacing))
    return diam


def normalize(nparray, min_v, max_v):
    nparray = np.where(nparray < min_v, min_v, nparray)
    nparray = np.where(nparray > max_v, max_v, nparray)
    nparray = (nparray + abs(min_v)) / (abs(min_v) + abs(max_v))
    return nparray


def get_zcenter_image_from_nparray(nparray):
    zcenter_im = nparray[int(nparray.shape[0]/2)]
    return zcenter_im


def get_zcenter_image_from_npyfile(npy):
    nparray = np.load(npy)
    zcenter_im = get_zcenter_image_from_nparray(nparray)
    return zcenter_im


def organize_data_to_subsets(data_dir):
    subsets = ['subset'+str(i) for i in range(10)]
    for subset in subsets:
        npys = file_io.get_files(data_dir, subset+'*')
        subset_dir = os.path.join(data_dir, subset)
        file_io.mkdirs(subset_dir)
        for old_file in npys:
            new_file = os.path.join(subset_dir, os.path.basename(old_file))
            os.rename(old_file, new_file)


def extract_data(data_dir, output_dir, regex, copy=True):
    subsets = get_subsets(data_dir)
    for subset in subsets:
        subset_files = file_io.get_files(subset, regex)
        subset = os.path.join(output_dir, os.path.basename(subset))
        file_io.mkdirs(subset)
        for file in subset_files:
            file_ = os.path.join(subset, os.path.basename(file))
            if copy:
                shutil.copyfile(file, file_)
            else:
                os.rename(file, file_)
    organize_data_to_subsets(output_dir)
