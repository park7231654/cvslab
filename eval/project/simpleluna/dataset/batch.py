# Author: Seunghyun Kim
# Date: 26 Aug 2018
# Last updated: 28 Aug 2018

import os
from glob import glob
import multiprocessing as mtp

import pandas as pd
import numpy as np
from tqdm import tqdm

from ..utils import file_io


class LunaBatchMaker2D:
    def __init__(self, data_dirs, output_dir):
        if type(data_dirs).__name__ != 'list':
            msg = "the type of 'data_dirs' must be 'list'."
            assert False, msg
        self._data_dirs = data_dirs
        self._output_dir = output_dir
        self._intermediate = os.path.join(output_dir, 'intermediate')

    def _get_binary_label(self, file):
        label = None
        if 'cls.0' in file:
            label = np.array([0])
        else:
            label = np.array([1])
        return label

    def _flatten_helper(self, df_spl):
        def _(file):
            npy = np.load(file)
            im = npy.reshape(-1)
            return im
        df_spl['img'] = df_spl['file'].apply(_)
        return df_spl

    def _get_flatten_ims(self, df):
        cpu_cores = mtp.cpu_count()
        df_split = np.array_split(df, cpu_cores)
        pool = mtp.Pool(cpu_cores)
        df = pd.concat(pool.map(self._flatten_helper, df_split))
        pool.close()
        pool.join()
        return df

    def _get_file(self, subset, targets):
        list_ = np.array([])
        for target in targets:
            sub_list_ = np.array(glob(os.path.join(target, subset, '*.npy')))
            list_ = np.concatenate((list_, sub_list_))
        return list_

    def _intermediate_batch(self, imsize, interval):
        file_io.mkdirs(self._intermediate)
        targets = self._data_dirs
        lengths = []
        for target in targets:
            subsets = file_io.get_dirs(target, 'subset*')
            lengths.append(len(subsets))
        subsets = ['subset' + str(i) for i in range(max(lengths))]
        for subset in tqdm(subsets):
            subset_files = self._get_file(subset, targets)
            subset_files = [subset_files[i:i + interval]
                            for i in range(0, len(subset_files), interval)]
            for i in range(len(subset_files)):
                df = pd.DataFrame()
                df['file'] = pd.Series(subset_files[i])
                df = self._get_flatten_ims(df)
                df['lbl'] = df['file'].apply(self._get_binary_label)
                df = df.drop(columns=['file']).values
                size = len(df)
                batch = np.ndarray([size, imsize + 1])
                for j in range(size):
                    batch[j] = np.concatenate((df[j][0], df[j][1]))
                intermediate_batch_name = subset + '_' + str(i) + '.npy'
                np.save(os.path.join(self._intermediate, intermediate_batch_name), batch)

    def make_batch(self, imsize=48*48):
        def get_batch(list_):
            bat = np.load(list_[0])
            for i in range(1, len(list_)):
                bat_ = np.load(list_[i])
                bat = np.concatenate((bat, bat_))
            return bat
        
        self._intermediate_batch(imsize, interval=10000)
        files = file_io.get_files(self._intermediate, '*')
        subsets = [file[:-4] for file in files]
        for i in range(len(subsets)):
            pardir = os.path.abspath(os.path.join(subsets[i], os.pardir))
            basename = os.path.basename(subsets[i])
            if '_' in basename:
                subsets[i] = os.path.join(pardir, basename.split('_')[0])
        subsets = list(set(subsets))
        subsets.sort()
        for subset in tqdm(subsets):
            subsetname = os.path.basename(subset)
            files = file_io.get_files(self._intermediate, subsetname+'*')
            batch = get_batch(files)
            np.save(os.path.join(self._output_dir, subsetname + '.npy'), batch)
            for file in files:
                os.remove(file)
        os.rmdir(self._intermediate)