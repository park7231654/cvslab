import os
import numpy as np

from keras.models import load_model
from keras.utils import to_categorical

from sklearn.metrics import roc_auc_score

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
    auc_storage = 0 # fixed
    best_epoch = 7 # LeNet-5 = 6 / VGG-16 = 19 / ResNet-152 = 9 / Inception-V3 = 2 / DenseNet-201 = 7 / NASNet = 11
    total_number_of_subset = 8 # LeNet-5 = 8 / VGG-16 = 9 / ResNet-152 = 7 / Inception-V3 = 7 / DenseNet-201 = 8 / NASNet = 10

    model_name = 'DenseNet-201' # LeNet-5 / VGG-16 / ResNet-152 / DenseNet-201 / NASNet
    model = load_model('LUNA16_2D_SUBSET0.h5')
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.summary()

    '''
    LeNet-5 = Subset0, Subset1, Subset4, Subset5, Subset6, Subset7, Subset8, Subset9
    VGG-16 = Subset0, Subset1, Subset3, Subset4, Subset5, Subset6, Subset7, Subset8, Subset9
    ResNet-152 = Subset0, Subset1, Subset4, Subset5, Subset6, Subset8, Subset9
    Inception-V3 = Subset0, Subset1, Subset5, Subset6, Subset7, Subset8, Subset9
    DenseNet-201 = Subset0, Subset1, Subset4, Subset5, Subset6, Subset7, Subset8, Subset9
    NASNet = All Use
    '''

    for fold in range(0, 10):

        '''
        Folders to exclude
        LeNet-5 = Subset2, Subset3
        VGG-16 = Subset2
        ResNet-152 = Subset2, Subset3, Subset7
        Inception-V3 = Subset2, Subset3, Subset4
        DenseNet-201 = Subset2, Subset3
        NASNet = X
        '''

        if fold == ? or fold == ? or fold == ?:
            continue
        else:
            model.load_weights('LUNA16_2D_SUBSET'+str(fold)+'.h5')

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
            y_test = to_categorical(class_label[:, 2304], 2)

            auc = roc_auc_score(y_test, x_test)
            print('\n')
            print('='*70)
            print('Area Under the Curve of Subset_'+str(fold)+' : %.3f' %(auc))
            print('='*70)
            print('\n')

            auc_storage += auc

    avg_auc = auc_storage/total_number_of_subset
    print(str(model_name)+' '+str(best_epoch)+' Epoch AUC : %.3f' %(avg_auc))


