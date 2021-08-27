# -*- coding=utf-8 -*-
import argparse
import os
import json
from skimage import io
from skimage.transform import resize

global_image_num = 0
char_set = set()


not_include=['■', '©', '、', '>', '_', '™', '=', '（', '▪', '✱', '¼', '▶', 'Ⅰ', '√', '°', 'á', '”', '{', '»', '：', '○', '}', '．', '～', '<', '…', '“', '◎', '▲', '†', '一', '∨', '×', '¾', '℃', '–', '\\', '＄', '／', '⠁', '》', 'ɪ', '¤', '☺', 'μ', '〕', '〔', '◆', '▼', '„', '˚', 'Í', '｜', '丨', '◇', '℉', '₂', '⁺', '；', '！', '☑', '※', '️', '①', 'Ⅱ', '⅓', 'ı', '③', '②', '→', '\xad', '？', '＆', '。', '✕', '④', '⅔', '฿', 'ʊ', '＝', '％', '✳', '✓', '☆', '‘', 'ə', 'ń', 'Ü', '§', '¥', '`', '《', '✶', '►', '∧', '↓', 'Ⅳ', 'Ⅲ', '⅛', '’', 'ł', 'İ', 'ę', 'ç', 'È', 'Å', 'À', '］', '－', '＇', '口', '」', '「', '♡', '◀', '┃', 'ⓥ', '⑮', '⑭', '⑬', '⑫', '⑪', '⑩', '⑨', '⑧', '⑦', '⑥', '⑤', '↑', '←', 'Ⅹ', 'Ⅷ', 'Ⅶ', 'Ⅵ', 'Ⅴ', '₁', '⁻', '⁶', 'й', 'β', 'Λ', 'ʃ', 'ǵ', 'ş', 'Ş', 'Ą', 'ø', 'ï', 'Ç', '´', '³', '«'
,'•','≥','●',"#"]  # --> add it all time
#,'Á','É','Ñ','Ó','Ú','à','â','ã','ä','è','ê','ì','î','ò','ô','ö','ú','ü','ā','œ','Й'
invalid_boxes=0
max_length=0
input_height,input_width=32,280   # Parameter
def extract_train_data(src_image_root_path, src_label_json_file, save_image_path, save_txt_path):
    global global_image_num, char_set,invalid_boxes,max_length,input_width,input_height

    with open(src_label_json_file, 'r', encoding='utf-8') as in_file:
        print("Label json file read \n")
        label_info_dict = json.load(in_file)
        with open(os.path.join(save_txt_path, 'train.txt'), 'a', encoding='utf-8') as out_file:
            print("Output train file Generated \n")
            for image_name, text_info_list in label_info_dict.items():

                src_image = io.imread(os.path.join(src_image_root_path, image_name))
                for text_info in text_info_list:
                    try:
                        flag=0
                        text = text_info['label']

                        if len(text)>150:
                            continue
                        
                        if len(text)>max_length:
                            max_length=len(text)
                        
                        for i in str(text):
                            if i in not_include:
                                flag=1
                        
                                
                        if flag==1:
                            continue
                            
                        src_point_list = text_info['points']
                        
                        image=src_image[round(src_point_list[0][1]):round(src_point_list[2][1]),round(src_point_list[0][0]):round(src_point_list[2][0]),:3]
                        input_channel=3   # Parameter
                        if image.size < 1:
                            continue


                        scale = image.shape[0] * 1.0 / input_height
                        image_width = int(image.shape[1] // scale)  #height evlo madangu scale panna required shape kidaikum nu paathutu athe madange width scale pandrom 
                                                                    #thus preserving the image size ratio
                        image = resize(image, (input_height,image_width),anti_aliasing=True))  
                        image_height, image_width = image.shape[0:2]
                        if image_width <= input_width:
                            new_image = np.ones((input_height, input_width, input_channel), dtype='uint8')
                            new_image[:] = 255
                            if input_channel == 1:
                                image = np.expand_dims(image, axis=2)
                            new_image[:, :image_width, :] = image    #if there are remaining space fill it with white.
                            image = new_image
                        else:
                            image = resize(image, (input_height,input_width),anti_aliasing=True)
                            if input_channel == 1:
                                image = np.expand_dims(image, axis=2)
                        crop_image = np.array(image, 'f') / 127.5 - 1.0


                        crop_image_name = '{}.jpg'.format(global_image_num)
                        global_image_num += 1
                        io.imsave(os.path.join(save_image_path, crop_image_name), crop_image,check_contrast=False)
                        out_file.write('{}\t{}\n'.format(crop_image_name, text))
                        
                        
                        text = text.replace('\r', '').replace('\n', '')
                        for char in text:
                            char_set.add(char)
                        
                        if global_image_num%1000==0:
                              print(global_image_num)
                    except:
                           invalid_boxes+=1
                           continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_train_image_path', type=str,
                        default='tmp_data/recognizer_images')
    parser.add_argument('--save_train_txt_path', type=str,
                        default='tmp_data/recognizer_txts')
    parser.add_argument('--train_image_common_root_path', type=str,
                        default='official_data/train_image_common')
    parser.add_argument('--common_label_json_file', type=str,
                        default='official_data/train_label_common.json')

    parser.add_argument('--train_image_special_root_path', type=str,
                       default='official_data/train_image_special')
    parser.add_argument('--special_label_json_file', type=str,
                       default='official_data/train_label_special.json')

    opt = parser.parse_args()

    save_train_image_path = opt.save_train_image_path
    save_train_txt_path = opt.save_train_txt_path

    train_image_common_root_path = opt.train_image_common_root_path
    common_label_json_file = opt.common_label_json_file
    
    extract_train_data(train_image_common_root_path,
                       common_label_json_file,
                       save_train_image_path,
                       save_train_txt_path)
    
    print('Common Image count : {}'.format(global_image_num))
    common=global_image_num
    
    train_image_special_root_path = opt.train_image_special_root_path
    special_label_json_file = opt.special_label_json_file

    extract_train_data(train_image_special_root_path,
                      special_label_json_file,
                      save_train_image_path,
                      save_train_txt_path)
    
    print("---****----")
    print("Special count : {}".format(global_image_num-common))
    print('Total Image num is {}.'.format(global_image_num))
        
    print("Invalid Boxes : ",invalid_boxes)

    print("Max Length",max_length)

    char_list = list(char_set)
    char_list.sort()

    with open('recognizer/tools/dictionary/chars.txt', 'a', encoding='utf-8') as out_file:
        for char in char_list:
            out_file.write('{}\n'.format(char))

'''bash
python recognizer/tools/extract_train_data.py
'''

