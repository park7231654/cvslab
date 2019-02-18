# Author: Seunghyun Kim
# Date: 13 Aug 2018


import os
from glob import glob


def _get_list(path, isdir, regex):
    list_ = glob(os.path.join(path, regex))
    if isdir:
        list_ = list(filter(os.path.isdir, list_))
    else:
        list_ = list(filter(os.path.isfile, list_))
    return list_


def get_dirs(path, regex='*'):
    return _get_list(path, True, regex)


def get_files(path, regex='*'):
    return _get_list(path, False, regex)


def mkdirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)