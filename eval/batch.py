import os
import pickle
from glob import glob

import numpy as np
from tqdm import tqdm


np.random.seed(42)


def save_batch(input_dir, output_dir):
    if os.path.isdir(output_dir):
        msg = 'The output directory already exists. ' + \
              'If you want to save the batch file again, ' + \
              'delete the output directory and try again.'
        assert False, msg
        
    os.makedirs(output_dir)
        
    subset_dirs = glob(os.path.join(input_dir, 'subset*'))
    subset_dirs.sort()
    
    minibatch_size = 1000
    
    for subset_dir in subset_dirs:
        npy_files = glob(os.path.join(subset_dir, '*.npy'))
        num_npy_files = len(npy_files)
        num_minibatches = int(np.ceil(num_npy_files / minibatch_size))
        
        buffer = []
        y = np.ndarray([num_npy_files,])
        for i in tqdm(range(num_minibatches)):
            stt = i * minibatch_size
            end = min((i + 1) * minibatch_size, num_npy_files)
            
            minibatch = np.load(npy_files[stt]).reshape(1, 48, 48)
            y[stt] = int(npy_files[stt].split('_')[1][-1])
            for j in range(stt+1, end):
                sample = np.load(npy_files[j]).reshape(1, 48, 48)
                minibatch = np.vstack((minibatch, sample))
                y[j] = int(npy_files[j].split('_')[1][-1])
                
            buffer.append(minibatch)
            
        x = buffer[0]
        for minibatch in tqdm(buffer):
            x = np.vstack((x, minibatch))
        
        indices = np.arange(num_npy_files)
        np.random.shuffle(indices)
        
        x, y = x[indices], y[indices]
            
        batch_name = os.path.basename(subset_dir) + '.npy'
        batch_path = os.path.join(output_dir, batch_name)
        
        with open(batch_path, 'wb') as f:
            pickle.dump((x, y), f)


def main():
    augment_dir = '/data/datasets/luna16-augment'
    batch_dir = '/data/datasets/luna16'
    
    save_batch(augment_dir, batch_dir)
    
    
if __name__ == '__main__':
    main()