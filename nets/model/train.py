import os
import numpy as np

from keras.utils import to_categorical
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

np.random.seed(0)

def data_generator(batches, minibatch_size):
    while True:
        for batch in batches:
            data = np.load(batch)
            np.random.shuffle(data)
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

if __name__ == '__main__':
    
    data_dir = 'LUNA16_2D'

    # Load Model or Model Define
    model = load_model()
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    for fold in range(0, 10):
       
        # Load Weights
        model.load_weights()
        
        # ModelCheckpoint Callback Function Define
        checkpoint = ModelCheckpoint(
            filepath='checkpoints/subset' + str(fold) + '_{epoch}.h5',
            monitor='loss',
            verbose=1,
            save_best_only=False,
            save_weights_only=False,
            mode='auto')

        minibatch_size = 100
        epochs = 10

        train_batches = [os.path.join(data_dir, 'subset'+str(i)+'.npy') for i in range(10) if i != fold]
        train_steps_per_epoch = get_steps_per_epoch(train_batches, minibatch_size)
        luna_train_generator = data_generator(train_batches, minibatch_size)

        # Model Training
        print('****Training****')
        model.fit_generator(luna_train_generator, steps_per_epoch = train_steps_per_epoch, epochs=epochs, callbacks=[checkpoint])
        print('****Finished****')
