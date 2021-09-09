
import argparse
import os


import tensorflow as tf

from recognizer.models.crnn_model import crnn_model_based_on_densenet_crnn_time_softmax_activate
from recognizer.tools.generator import Generator
from recognizer.tools.config import config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_save_dir',
                        default='checkpoint/recognizer')
    parser.add_argument('--log_dir',
                        default='output/recognizer_log')
    parser.add_argument('--train_image_dir',
                        default='tmp_data/recognizer_images/train')
    parser.add_argument('--test_image_dir',
                        default='tmp_data/recognizer_images/test')                      
    parser.add_argument('--txt_dir',
                        default='tmp_data/recognizer_txts')

    parser = parser.parse_args()
    
    batch_size = 128
    max_label_length = 280
    epochs = 12
    base_model, model = crnn_model_based_on_densenet_crnn_time_softmax_activate(initial_learning_rate=0.0005)

    train_loader = Generator(root_path=parser.train_image_dir,
                             input_map_file=os.path.join(parser.txt_dir, 'real_train.txt'),
                             batch_size=batch_size,
                             max_label_length=max_label_length,
                             input_shape=(32, 280, 3),
                             is_training=True)

    valid_loader = Generator(root_path=parser.test_image_dir,
                            input_map_file=os.path.join(parser.txt_dir, 'real_test.txt'),
                            batch_size=batch_size,
                            max_label_length=max_label_length,
                            input_shape=(32, 280, 3),
                            is_training=True)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(parser.model_save_dir, 'weights_crnn-{epoch:03d}-{loss:.3f}.h5'),#add accuracy too
        monitor='loss', save_best_only=False, save_weights_only=True
    )
    change_learning_rate = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=1)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=7, verbose=1)
    tensor_board = tf.keras.callbacks.TensorBoard(log_dir=parser.log_dir, write_graph=True)
    model.fit_generator(generator=train_loader.__next__(),
                        steps_per_epoch=train_loader.num_samples() // batch_size,
                        validation_data=valid_loader.__next__(),
                        validation_steps=valid_loader.num_samples()//batch_size,
                        epochs=epochs,
                        initial_epoch=0,
                        callbacks=[checkpoint,change_learning_rate, tensor_board, early_stop])


