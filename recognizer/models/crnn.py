import tensorflow as tf
from recognizer.tools.config import config
from tensorflow.keras.layers import Conv2D,Dense,TimeDistributed,Flatten,Reshape



def densenet_crnn_time(inputs):
    densenet = tf.keras.applications.DenseNet121(input_tensor=inputs, include_top=False, weights=None)
    x = Conv2D(filters=512, kernel_size=(5, 5), strides=(2, 1), padding='same')(densenet.layers[50].output)
    #x = Reshape((160, 1, 896))(x)
    x = Reshape((280, 1, 512))(x)
    x = TimeDistributed(Flatten())(x)
    x = Dense(config.num_class, activation=None)(x)
    return x
