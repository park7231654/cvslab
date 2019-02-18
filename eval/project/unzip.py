# Author: Seunghyun Kim
# Date: 17 Feb 2019
# Last updated: 17 Feb 2019


import os
from glob import glob


"""
Unzip all data.

We assume that you have downloaded all the necessary data from
the [LUNA16](https://luna16.grand-challenge.org) website. This
task requires at least 120 Gb of free space and 7-zip package.
If you see that the command is not found when you run the task,
see the following URL: https://www.7-zip.org/
"""

# The directory 'origin_dir' contains the following zip files
# : subset0.zip, ..., subset9.zip, candidates_V2.csv, and CSVFILES.zip

data_dir = 'ORIGINAL_DATA_DIR'
unzip_dir = 'OUTPUT_DIR'

if not os.path.isdir(unzip_dir):
    os.makedirs(unzip_dir)
    
zip_files = glob(os.path.join(data_dir, '*.zip'))
zip_files.sort()

for zip_file in tqdm(zip_files):
    os.system('7z x ' + zip_file + ' -o' + unzip_dir + ' -aos')