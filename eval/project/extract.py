# Author: Seunghyun Kim
# Date: 17 Feb 2019
# Last updated: 17 Feb 2019

import os

from simpleluna.dataset.preprocess import LunaNoduleCropper


resample_dir = 'RESAMPLE_DIR'
extract_dir = 'EXTRACT_DIR'

crop_size = 56
crop_margin = 0
wrap_size = 56

voxel_csv = os.path.join(resample_dir, 'voxel_candidates_V2.csv')

"""
Output details:
name, class-1, class-0
----------------------
subset0, 138, 78997
subset1, 170, 70842
subset2, 181, 74277
subset3, 158, 75792
subset4, 170, 76413
subset5, 127, 75564
subset6, 154, 76517
subset7, 120, 74943
subset8, 195, 74293
subset9, 144, 75780
----------------------
total, 1557, 753418
19.1 GB
"""
noduleCropper = LunaNoduleCropper(resample_dir, extract_dir, voxel_csv)
noduleCropper.set_cropping(crop_size, crop_margin, check=False)
noduleCropper.set_wrapping(wrap_size)
noduleCropper.run(get_patch=True)