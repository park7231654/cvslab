from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def LeNet5(input_tensor=None, input_shape=None):
    img_input = Input(shape=input_shape)
    x = Conv2D(32,(3, 3), activation='relu',name='block1_conv1')(input_shape)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    x = Conv2D(64,(3,3),activation='relu',name='block1_conv2')
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(64, activation='relu', name='fc1')(x)
    x = Dense(2, activation='softmax', name='prediction')(x)

    model = Model(img_input, x, name='lenet5')

    return model

