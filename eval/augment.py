# Author: Seunghyun Kim
# Date: 19 Feb 2019
# Last updated: 20 Feb 2019
# --- Ad hoc ---

import os
from glob import glob
from enum import Enum

import numpy as np
from imgaug import augmenters as iaa
from tqdm import tqdm


Flip = Enum('Flip', ['NONE', 'UP_DOWN', 'LEFT_RIGHT'])
Crop = Enum('Crop', ['CENTER', 'NORTH_WEST', 'SOUTH_EAST',
                     'NORTH_EAST', 'SOUTH_WEST'])


def _add_tag(name, tag):
    if name[-4:] == '.npy':
        name = name[:-4]
    name += tag + '.npy'
    return name


def rescale(img, name, scale):
    seq = iaa.Sequential([iaa.Affine(scale=scale)])
    rescale_img = seq.augment_image(img)
    tag = '_rsc' + str(scale)
    tagged_name = _add_tag(name, tag)
    return rescale_img, tagged_name


def flip(img, name, direction):
    tag = '_flp'
    if direction == Flip.NONE:
        tag += 'no'
        flipped_img = img
    elif direction == Flip.UP_DOWN:
        tag += 'ud'
        seq = iaa.Sequential([iaa.Flipud(1.0)])
        flipped_img = seq.augment_image(img)
    elif direction == Flip.LEFT_RIGHT:
        tag += 'lr'
        seq = iaa.Sequential([iaa.Fliplr(1.0)])
        flipped_img = seq.augment_image(img)
    tagged_name = _add_tag(name, tag)
    return flipped_img, tagged_name


def rotate(img, name, angle):
    seq = iaa.Sequential([iaa.Affine(rotate=angle, mode='edge')])
    rotated_img = seq.augment_image(img)
    tag = '_rot' + str(angle).zfill(3)
    tagged_name = _add_tag(name, tag)
    return rotated_img, tagged_name


def crop(img, name, position, size):
    height, width = img.shape[0], img.shape[1]
    tag = '_crp'
    if position == Crop.CENTER:
        tag += 'ct'
        cropped_img = img[int((height/2)-(size/2)):int((height/2)-(size/2))+size,
                          int((width/2)-(size/2)):int((width/2)-(size/2))+size]
    elif position == Crop.NORTH_WEST:
        tag += 'nw'
        cropped_img = img[:size, :size]
    elif position == Crop.SOUTH_EAST:
        tag += 'se'
        cropped_img = img[height-size:size+height, width-size:size+width]
    elif position == Crop.NORTH_EAST:
        tag += 'ne'
        cropped_img = img[:size, width-size:(width-size)+size]
    elif position == Crop.SOUTH_WEST:
        tag += 'sw'
        cropped_img = img[height-size:(height-size)+size, :size]
    tagged_name = _add_tag(name, tag)
    return cropped_img, tagged_name


def augment(extract_dir, augment_dir):
    if os.path.isdir(augment_dir):
        msg = 'The output directory already exists. ' + \
              'If you want to augment again, delete ' + \
              'the output directory and try again.'
        assert False, msg
    
    # arguments(hard-code)
    scales = [1.0, 1.3, 1.6, 1.9]
    angles = [i * 45 for i in range(8)]
    direcs = [direc for direc in Flip]
    posits = [posit for posit in Crop]
    size = 48
    
    subset_dirs = glob(os.path.join(extract_dir, 'subset*'))
    subset_dirs.sort()
    
    for subset_dir in subset_dirs:
        subset_name = os.path.basename(subset_dir)
        new_subset_path = os.path.join(augment_dir, subset_name)
        os.makedirs(new_subset_path)
        
        npy_files = glob(os.path.join(subset_dir, '*.npy'))
        for npy_file in tqdm(npy_files):
            img = np.load(npy_file)
            name = os.path.basename(npy_file)
            
            # Positive images.
            if 'cls1' in npy_file:
                for scale in scales:
                    rsc_img, rsc_name = rescale(img, name, scale)
                    for direc in direcs:
                        flp_img, flp_name = flip(rsc_img, rsc_name, direc)
                        for angle in angles:
                            rot_img, rot_name = rotate(flp_img, flp_name, angle)
                            for posit in posits:
                                crp_img, crp_name = crop(rot_img, rot_name, posit, size)
                                path = os.path.join(new_subset_path, crp_name)
                                np.save(path, crp_img)
                                
            # Negative images.
            elif 'cls0' in npy_file:
                rsc_img, rsc_name = rescale(img, name, 1.0)
                flp_img, flp_name = flip(rsc_img, rsc_name, Flip.NONE)
                rot_img, rot_name = rotate(flp_img, flp_name, 0)
                crp_img, crp_name = crop(rot_img, rot_name, Crop.CENTER, size)
                path = os.path.join(new_subset_path, crp_name)
                np.save(path, crp_img)

        
def main():
    # extract_dir:
    #   The directory path that contains the last output of preprocessing.py.
    #
    # augment_dir:
    #   The directory path where augmented candidates will be stored.
    extract_dir = '/data/datasets/luna16-extract'
    augment_dir = '/data/datasets/luna16-augment'

    # Run augment().
    augment(extract_dir, augment_dir)
    
    
if __name__ == '__main__':
    main()