# -*- coding=utf-8 -*-

import tensorflow as tf

from recognizer.models.crnn import densenet_crnn_time

K = tf.compat.v1.keras.backend
Lambda = tf.keras.layers.Lambda
Input = tf.keras.layers.Input


def ctc_lambda_func(args):
    prediction, label, input_length, label_length = args
    return K.ctc_batch_cost(label, prediction, input_length, label_length)

class CERMetric(tf.keras.metrics.Metric):
    """
    A custom Keras metric to compute the Character Error Rate
    """
    def __init__(self, name='CER_metric', **kwargs):
        super(CERMetric, self).__init__(name=name, **kwargs)
        self.cer_accumulator = self.add_weight(name="total_cer", initializer="zeros")
        self.counter = self.add_weight(name="cer_count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        input_shape = K.shape(y_pred)
        input_length = tf.ones(shape=input_shape[0]) * K.cast(input_shape[1], 'float32')

        decode, log = K.ctc_decode(y_pred,
                                    input_length,
                                    greedy=True)

        decode = K.ctc_label_dense_to_sparse(decode[0], K.cast(input_length, 'int32'))
        y_true_sparse = K.ctc_label_dense_to_sparse(y_true, K.cast(input_length, 'int32'))

        decode = tf.sparse.retain(decode, tf.not_equal(decode.values, -1))
        distance = tf.edit_distance(decode, y_true_sparse, normalize=True)

        self.cer_accumulator.assign_add(tf.reduce_sum(distance))
        self.counter.assign_add(len(y_true))

    def result(self):
        return tf.math.divide_no_nan(self.cer_accumulator, self.counter)

    def reset_states(self):
        self.cer_accumulator.assign(0.0)
        self.counter.assign(0.0)


class WERMetric(tf.keras.metrics.Metric):
    """
    A custom Keras metric to compute the Word Error Rate
    """
    def __init__(self, name='WER_metric', **kwargs):
        super(WERMetric, self).__init__(name=name, **kwargs)
        self.wer_accumulator = self.add_weight(name="total_wer", initializer="zeros")
        self.counter = self.add_weight(name="wer_count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        input_shape = K.shape(y_pred)
        input_length = tf.ones(shape=input_shape[0]) * K.cast(input_shape[1], 'float32')

        decode, log = K.ctc_decode(y_pred,
                                    input_length,
                                    greedy=True)

        decode = K.ctc_label_dense_to_sparse(decode[0], K.cast(input_length, 'int32'))
        y_true_sparse = K.ctc_label_dense_to_sparse(y_true, K.cast(input_length, 'int32'))

        decode = tf.sparse.retain(decode, tf.not_equal(decode.values, -1))
        distance = tf.edit_distance(decode, y_true_sparse, normalize=True)
        
        correct_words_amount = tf.reduce_sum(tf.cast(tf.not_equal(distance, 0), tf.float32))

        self.wer_accumulator.assign_add(correct_words_amount)
        self.counter.assign_add(len(y_true))

    def result(self):
        return tf.math.divide_no_nan(self.wer_accumulator, self.counter)

    def reset_states(self):
        self.wer_accumulator.assign(0.0)
        self.counter.assign(0.0)

def crnn_model_based_on_densenet_crnn_time_softmax_activate(initial_learning_rate=0.0005):
    
    
    
    shape = (32, 280, 3)
    inputs = tf.keras.layers.Input(shape=shape, name='input_data')
    output = densenet_crnn_time(inputs=inputs, activation='softmax')
    model_body = tf.keras.models.Model(inputs, output)
    model_body.summary()

    label = Input(name='label', shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([output, label, input_length, label_length])

    model = tf.keras.models.Model(inputs=[inputs, label, input_length, label_length], outputs=loss_out)
  #  cer=CERMETRIC()
   # wer=WERMETRIC()
    model.compile(loss={'ctc': lambda y_true, prediction: prediction},
                  optimizer=tf.keras.optimizers.Adam(initial_learning_rate), metrics=['accuracy',"CERMetric","WERMetric"])
    #sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)  -> other optimizer
    return model_body, model


