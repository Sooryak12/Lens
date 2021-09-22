import os
import json
import cv2
from skimage import io
import numpy as np
from itertools import groupby

from recognizer.models.crnn_model import crnn_model_mobile_net
from recognizer.tools.config import config
from recognizer.tools.utils import get_dict

os.environ["CUDA_VISIBLE_DEVICES"] = "0"




def load_model(model_type, weight):
        base_model, _ = crnn_model_mobile_net()
        base_model.load_weights(weight)
    return base_model


def predict(image, input_shape, base_model):
    input_height, input_width, input_channel = input_shape
    scale = image.shape[0] * 1.0 / input_height
    image_width = int(image.shape[1] // scale)
    if image_width <= 0:
        return ''
    image = cv2.resize(image, (image_width, input_height))
    image_height, image_width = image.shape[0:2]
    if image_width <= input_width:
        new_image = np.ones((input_height, input_width, input_channel), dtype='uint8')
        new_image[:] = 255
        if input_channel == 1:
            image = np.expand_dims(image, axis=2)
        try:
          new_image[:, :image_width, :] = image
        except:
          image =np.expand_dims(image,axis=2)
          image= np.repeat(image, 3, axis=2)
          new_image[:, :image_width, :] = image    
        image = new_image
    else:
        image = cv2.resize(image, (input_width, input_height))
    text_image = np.array(image, 'f') / 127.5 - 1.0
    try:
      text_image = np.reshape(text_image, [1, input_height, input_width, input_channel])
    except:   
        text_image=np.expand_dims(text_image,axis=2)   
        text_image= np.repeat(text_image, 3, axis=2)
        text_image = np.reshape(text_image, [1, input_height, input_width, input_channel])
    # y_pred=0
    # return y_pred
    y_pred = base_model.predict(text_image)
    y_pred = y_pred[:, :, :]
    char_list = list()
    pred_text = list(y_pred.argmax(axis=2)[0])
    for index in groupby(pred_text):
        if index[0] != config.num_class - 1:
            char_list.append(character_map_table[str(index[0])])
    return u''.join(char_list)


if __name__ == '__main__':
    
    char_path=config.char_path
    model_path='/content/weights_crnn-010-12.184.h5'
    null_json_path='null_submission_non_max.json'
    test_image_path='offical_data/test_image'
    submission_path='output/test_submission.json'


    character_map_table = get_dict(char_path)
    input_shape = (32, 280, 3)
    model = load_model(model_path)
    print(' model loaded')

    test_label_json_file = null_json_path
    test_image_root_path = test_image_path
    with open(test_label_json_file, 'r', encoding='utf-8') as in_file:
        label_info_dict = json.load(in_file)
        for idx, info in enumerate(label_info_dict.items()):
            image_name, text_info_list = info
            src_image = io.imread(f"official_data/test_image/{image_name}")
            if idx %25==0:
                print(idx)
            for index, text_info in enumerate(text_info_list):
                try:
                    src_point_list = text_info['points']
                    try:
                      crop_image=src_image[round(src_point_list[0][1]):round(src_point_list[2][1]),round(src_point_list[0][0]):round(src_point_list[2][0]),:3]
                    except:
                      crop_image=src_image[round(src_point_list[0][1]):round(src_point_list[2][1]),round(src_point_list[0][0]):round(src_point_list[2][0])]
                    rec_result = predict(crop_image, input_shape, model)
                    text_info['label'] = rec_result
                except Exception as e:
                    print(f"{image_name} : {index} -> {e}")

    save_label_json_file = submission_path
    with open(save_label_json_file, 'w',encoding="utf-8") as out_file:
        out_file.write(json.dumps(label_info_dict))

