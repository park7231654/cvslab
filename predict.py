import os
import numpy as np
import pandas as pd

from keras.models import load_model
from keras.utils import to_categorical

np.random.seed(0)

def test_data_generator(batches, minibatch_size):
    while True:
        for batch in batches:
            data = np.load(batch)
            samples = data.shape[0]
            minibatches = samples // minibatch_size
            if samples % minibatch_size > 0:
                minibatches += 1
            for minibatch in range(minibatches):
                section = slice(minibatch * minibatch_size,
                                (minibatch + 1) * minibatch_size)
                x_train = data[:, :2304][section].reshape(-1, 48, 48, 1)
                y_train = to_categorical(data[:, 2304][section], 2)
                yield (x_train, y_train) # yield(x_train, y_train)

def get_steps_per_epoch(batches, minibatch_size):
    print('Calc steps_per_epoch ...')
    total_step = 0
    for batch in batches:
        samples = np.load(batch).shape[0]
        minibatches = samples // minibatch_size
        if samples % minibatch_size > 0:
            minibatches += 1
        print(os.path.basename(batch), ':', minibatches)
        total_step += minibatches
    print('Total step_per_epoch :', total_step)
    print('Done.')
    return total_step

if __name__ == "__main__":
    
    data_dir = 'LUNA16_2D'
    model_name = ? # LeNet-5, VGG-16, ResNet-152, Inception-V3, DenseNet-201, NASNet

    # Load Model or Model Define
    model = load_model()
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.summary()

    for epoch_index in range(1, 11):

        for fold in range(0, 10):
               
            # Load Weights
            model.load_weights()

            val_batches = [os.path.join(data_dir, 'subset'+str(fold)+'.npy')]
            test_batches= os.path.join(data_dir, 'subset'+str(fold)+'.npy') # For extracting test data set labels

            minibatch_size = 100

            val_steps_per_epoch = get_steps_per_epoch(val_batches, minibatch_size)
            luna_val_generator = test_data_generator(val_batches, minibatch_size)

            # Model Predict
            print('****Prediction****')
            x_test = model.predict_generator(luna_val_generator, steps = val_steps_per_epoch, verbose=1)
            print('****Finished****')

            # Extraction for Test Data Set Label
            class_label = np.load(test_batches)
            y_test = class_label[:, 2304]

            prediction = np.concatenate((x_test, y_test.reshape(-1,1)), axis=1)

            # Submit Test Results
            submission = pd.DataFrame(data = prediction, columns=list(['acc_0','acc_1','y']))
            submission.to_csv(
                path_or_buf= str(model_name)+'_'+str(epoch_index)+'_'+'Epoch_Test_'+str(fold)+'.csv',
                header = True,
                index = False)
