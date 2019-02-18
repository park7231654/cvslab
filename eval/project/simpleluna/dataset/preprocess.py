# Author: Seunghyun Kim
# Date: 24 Aug 2018
# Last updated: 26 Aug 2018


import os
import shutil

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from ..utils import file_io
from ..utils import luna_io
from ..utils import node_io


class LunaMHDResampleNormalizer:

    def __init__(self, data_dir, output_dir, csv_file, norm_min=-1000, norm_max=400):
        self._luna_dir = data_dir
        self._output_dir = output_dir
        self._csv_file = csv_file
        self._min = norm_min
        self._max = norm_max

    def run(self, only_csv=False):
        file_io.mkdirs(self._output_dir)
        subsets = luna_io.get_subsets(self._luna_dir)
        mhds = luna_io.get_files_from_subsets(subsets, regex='*.mhd')
        df = luna_io.get_mhd_dataframe(mhds, self._csv_file)
        df['vcoordX'] = None
        df['vcoordY'] = None
        df['vcoordZ'] = None
        if 'diameter_mm' in df.keys():
            df['vdiameter'] = None
        df['npy'] = None

        for mhd in tqdm(mhds):
            df_mhd = df[df['file'] == mhd]
            if df_mhd.shape[0] > 0:
                subset = os.path.abspath(os.path.join(mhd, os.pardir))
                subset = os.path.basename(subset)
                seriesuid = str(os.path.basename(mhd)[:-4].split('.')[-1])
                filename = subset + '_' + seriesuid + '.npy'
                filepath = os.path.join(self._output_dir, filename)
                
                image = sitk.ReadImage(mhd)
                array = sitk.GetArrayFromImage(image)
                origin = np.array(image.GetOrigin())[::-1]
                spacing = np.array(image.GetSpacing())[::-1]
                
                if only_csv:
                    _, spacing = luna_io.resample(array, spacing, only_spacing=True)
                else:
                    array, spacing = luna_io.resample(array, spacing)
                    array = luna_io.normalize(array, self._min, self._max)
                    np.save(filepath, array)
                    
                for i, row in df_mhd.iterrows():
                    center = np.array([row.coordZ, row.coordY, row.coordX])
                    vcenter = np.rint((center - origin) / spacing)
                    df.at[i, 'vcoordX'] = vcenter[2]
                    df.at[i, 'vcoordY'] = vcenter[1]
                    df.at[i, 'vcoordZ'] = vcenter[0]
                    if 'diameter_mm' in df.keys():
                        vdiam = max(np.rint(row['diameter_mm'] / spacing))
                        df.at[i, 'vdiameter'] = vdiam
                    df.at[i, 'npy'] = filename
                    
        if only_csv:
            pass
        else:
            luna_io.organize_data_to_subsets(self._output_dir)
        new_csv_file = 'voxel_' + os.path.basename(self._csv_file)
        new_csv_file = os.path.join(self._output_dir, new_csv_file)
        df = df.drop(['seriesuid', 'file'], axis=1)
        df.to_csv(new_csv_file)


class LunaNoduleCropper:

    def __init__(self, data_dir, output_dir, voxel_csv_file):
        
        self._vdata_dir = data_dir
        self._output_dir = output_dir
        self._csv_file = voxel_csv_file
        self._df = luna_io.get_dataframe(voxel_csv_file)
        self._crop = {'size': None, 'margin': 0}
        self._wrap = {'size': None}
        self._err = {
            0: "margin must be greater than or equal to 0.",
            1: "vdiameter must be included in voxel_csv_file if size == None.",
            2: "size must be greater than or equal to 0.",
            3: "must be greater than or equal largest vdiam.",
            4: "size must be greater than or equal manual crop size.",
            5: "vdiameter must be included in voxel_csv_file if check == True."
        }

    def set_cropping(self, size=None, margin=0, check=False):
        
        if margin < 0:
            assert False, self._err[0]
        self._crop['margin'] = margin
        
        if size is None:
            if 'vdiameter' not in self._df.keys():
                assert False, self._err[1]
                
        else:
            if size < 0:
                assert False, self._err[2]
            self._crop['size'] = np.array([size] * 3)
            if check:
                if 'vdiameter' not in self._df.keys():
                    assert False, self._err[5]
                idx = self._df['vdiameter'].idxmax()
                largest_vdiam = self._df.ix[idx][['vdiameter']]
                if size + (margin * 2) < largest_vdiam:
                    prefix = "size + margin * 2 "
                    info = " (largest vdiam = %s)" % largest_vdiam
                    assert False, prefix + self._err[3] + info

    def set_wrapping(self, size):
        
        if size <= 0:
            assert False, self._err[2]
            
        if self._crop['size'] is None:
            idx = self._df['vdiameter'].idxmax()
            largest_vdiam = self._df.ix[idx][['vdiameter']]
            if size < largest_vdiam:
                prefix = 'size'
                info = " (largest vdiam = %s)" % largest_vdiam
                assert False, prefix + self._err[3] + info
                
        else:
            crop_size = max(self._crop['size'] + self._crop['margin'])
            if size < crop_size:
                info = " (manual crop size = %s)" % crop_size
                assert False, self._err[4]
                
        self._wrap['size'] = np.array([size] * 3)

    def run(self, get_patch=False):
        
        file_io.mkdirs(self._output_dir)
        subsets = luna_io.get_subsets(self._vdata_dir)
        npys = luna_io.get_files_from_subsets(subsets, regex='*.npy')
        
        for npy in tqdm(npys):
            
            npyname = os.path.basename(npy)
            npypath = os.path.join(self._output_dir, npyname[:-4])
            
            array = np.load(npy)
            df_npy = self._df[self._df['npy'] == npyname]
            
            for i, row in df_npy.iterrows():
                
                nodepath = npypath + '_row.' + str(i)
                if 'class' in self._df.keys():
                    nodepath += '_cls.' + str(row['class'])
                nodepath += '.npy'
                
                vcenter = np.array([row.vcoordZ, row.vcoordY, row.vcoordX])
                crop_size = self._crop['size']
                crop_margin = self._crop['margin']
                if crop_size is None:
                    crop_size = np.array([row.vdiameter] * 3)
                node_array = node_io.crop(array, vcenter, crop_size, crop_margin)
                wrap_size = self._wrap['size']
                if wrap_size is not None:
                    node_array = node_io.wrap(node_array, wrap_size)
                if get_patch:
                    node_array = luna_io.get_zcenter_image_from_nparray(node_array)
                    
                np.save(nodepath, node_array)
                
        luna_io.organize_data_to_subsets(self._output_dir)


class LunaNodulePicker:
    
    def __init__(self, data_dir, output_dir, regex):
        
        self._data_dir = data_dir
        self._output_dir = output_dir
        self._regex = regex

    def counting(self):
        
        count = []
        subsets = luna_io.get_subsets(self._data_dir)
        for subset in subsets:
            count.append(len(file_io.get_files(subset, self._regex)))
        total = 0
        for num in count:
            total += num
        count.append(total)
        return count

    def move(self, copy=True):
        
        file_io.mkdirs(self._output_dir)
        subsets = luna_io.get_subsets(self._data_dir)
        files = luna_io.get_files_from_subsets(subsets, self._regex)
        for file in tqdm(files):
            file_ = os.path.join(self._output_dir, os.path.basename(file))
            if copy:
                shutil.copyfile(file, file_)
            else:
                shutil.move(file, file_)
        luna_io.organize_data_to_subsets(self._output_dir)
