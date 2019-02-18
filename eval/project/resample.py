# Author: Seunghyun Kim
# Date: 17 Feb 2019
# Last updated: 17 Feb 2019

import os

from simpleluna.dataset.preprocess import LunaMHDResampleNormalizer


unzip_dir = 'UNZIP_DIR'
resample_dir = 'RESAMPLE_DIR'

csv = os.path.join(unzip_dir, 'candidates_V2.csv')
norm_min = -1000
norm_max = 400

resampleNormalizer = LunaMHDResampleNormalizer(unzip_dir, resample_dir,
                                               csv, norm_min, norm_max)
resampleNormalizer.run()