import tensorflow as tf
from recognizer.tools.config import config
from tensorflow.keras.layers import Conv2D,Dense,TimeDistributed,Flatten,Reshape,Input,Lambda
from recognizer.model.crnn import densenet_crnn_time

K = keras.backend



def densenet_crnn_time(inputs,activation=None):
    densenet = tf.keras.applications.DenseNet121(input_tensor=inputs, include_top=False, weights=None)
    x = Conv2D(filters=512, kernel_size=(5, 5), strides=(2, 1), padding='same')(densenet.layers[50].output)
    #x = Reshape((160, 1, 896))(x)
    x = Reshape((280, 1, 512))(x)
    x = TimeDistributed(Flatten())(x)
    x = Dense(config.num_class, activation=activation)(x)
    return x


def ctc_lambda_func(args):
    prediction, label, input_length, label_length = args
    return K.ctc_batch_cost(label, prediction, input_length, label_length)


def crnn_model_mobile_net(initial_learning_rate=0.0005):
    
    
    shape = (32, 280, 3)
    inputs = tf.keras.layers.Input(shape=shape)
    output = densenet_crnn_time(inputs=inputs, activation='softmax')
    model_body = tf.keras.models.Model(inputs, output)
    model_body.summary()

    label = Input(name='label', shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([output, label, input_length, label_length])

    model = tf.keras.models.Model(inputs=[inputs, label, input_length, label_length], outputs=loss_out)


    
    model.compile(loss={'ctc': lambda y_true, prediction: prediction},
                   optimizer=tf.keras.optimizers.Adam(initial_learning_rate), metrics=['accuracy'])

    # model.compile(loss={'ctc': lambda y_true, prediction: prediction},
    #                metrics={'ctc':lambda output,label :output}
    #              optimizer=tf.keras.optimizers.Adam(initial_learning_rate))
    #sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)  -> other optimizer
    return model_body, model

















## Ignore





"""
from itertools import groupby

    numpy version
    def numpyv(output,label):
        y_pred = output[:, :, :]
        char_list = list()
        prediction=[]
        for i in list(y_pred.argmax(axis=2)):
            pred_text = list(i)
            for index in groupby(pred_text):
                if index[0] != config.num_class - 1:
                    char_list.append(character_map_table[str(index[0])])
            predicton.append(u''.join(char_list))

    #tensorflow to numpy version
    def tf_to_numpy(output,label):
        output=output.numpy()
        y_pred = output[:, :, :]
        char_list = list()
        prediction=[]
        for i in list(y_pred.argmax(axis=2)):
            pred_text = list(i)
            for index in groupby(pred_text):
                if index[0] != config.num_class - 1:
                    char_list.append(character_map_table[str(index[0])])
            predicton.append(u''.join(char_list))


    def tf_v(output,label):
        y_pred = output[:, :, :]
        char_list = list()
        prediction=[]
        for i in list(tf.math.argmax(y_pred,axis=2)):
            pred_text = list(i)
            for index in groupby(pred_text):
                if index[0] != config.num_class - 1:
                    char_list.append(character_map_table[str(index[0])])
            predicton.append(u''.join(char_list))
"""