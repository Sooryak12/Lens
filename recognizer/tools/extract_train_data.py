# -*- coding=utf-8 -*-
import argparse
import os
import json
from skimage import io

global_image_num = 0
char_set = set()

not_include=['%', '·', '₹', 'é', '@', '★', '，', '?', 'í', '~', ';', '—', '€', '®', 'ñ', ']', '[', '½', '￡', 'ó', '）', '¢', '□', '￥', '■', '©', '、', '>', '_', '™', '=', '（', '▪', '✱', '¼', '▶', 'Ⅰ', 'á', '√', '°', '”', '{', '»', '：', '○', '}', '．', '～', '<', '…', '“', '◎', '▲', '†', '一', '∨', '×', '¾', '℃', '–', '\\', '＄', '／', '⠁', '》', 'ɪ', '¤', '☺', 'μ', '〕', '〔', '◆', '▼', '„', '˚', 'Í', '｜', '丨', '◇', '℉', '₂', '⁺', '；', '！', '☑', '①', '※', '️', 'Ⅱ', '⅓', 'ı', '③', '②', '→', '\xad', '？', '＆', '。', '✕', '④', '⅔', '฿', 'ʊ', '＝', '％', '✳', '✓', '☆', '‘', 'ə', 'ń', 'Ü', '§', '¥', '`', '《', '✶', '►', '∧', '↓', 'Ⅳ', 'Ⅲ', '⅛', '’', 'ł', 'İ', 'ę', 'ç', 'È', 'Å', 'À', '］', '－', '＇', '口', '」', '「', '♡', '◀', '┃', 'ⓥ', '⑮', '⑭', '⑬', '⑫', '⑪', '⑩', '⑨', '⑧', '⑦', '⑥', '⑤', '↑', '←', 'Ⅹ', 'Ⅷ', 'Ⅶ', 'Ⅵ', 'Ⅴ', '₁', '⁻', '⁶', 'й', 'β', 'Λ', 'ʃ', 'ǵ', 'ş', 'Ş', 'Ą', 'ø', 'ï', 'Ç', '´', '³', '«']
invalid_boxes=0

def extract_train_data(src_image_root_path, src_label_json_file, save_image_path, save_txt_path):
    global global_image_num, char_set,i
    i=0
    with open(src_label_json_file, 'r', encoding='utf-8') as in_file:
        print("Label json file read \n")
        label_info_dict = json.load(in_file)
        with open(os.path.join(save_txt_path, 'train.txt'), 'a', encoding='utf-8') as out_file:
            print("Output train file Generated \n")
            for image_name, text_info_list in label_info_dict.items():
                i=i+1
                if i%1000 ==0 :
                    print(i,end=" ")
                src_image = io.imread(os.path.join(src_image_root_path, image_name))
                for text_info in text_info_list:
                    try:
                        flag=0
                        text = text_info['label']
                        
                        for i in str(text):
                            if i in not_include:
                                flag=1
                                
                        if flag==1:
                            continue
                            
                        src_point_list = text_info['points']
                        
                        crop_image=src_image[round(src_point_list[0][1]):round(src_point_list[2][1]),round(src_point_list[0][0]):round(src_point_list[2][0]),:3]

                        if crop_image.size == 0:
                            continue
                        crop_image_name = '{}.jpg'.format(global_image_num)
                        global_image_num += 1
                        cv2.imwrite(os.path.join(save_image_path, crop_image_name), crop_image)
                        out_file.write('{}\t{}\n'.format(crop_image_name, text))
                        
                        
                        text = text.replace('\r', '').replace('\n', '')
                        for char in text:
                            char_set.add(char)
                        
                        if global_image_num%1000==0:
                              print(global_image_num)
                    except:
                           invalid_boxes+=1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_train_image_path', type=str,
                        default='tmp_data/recognizer_images')
    parser.add_argument('--save_train_txt_path', type=str,
                        default='tmp_data/recognizer_txts')
    parser.add_argument('--train_image_common_root_path', type=str,
                        default='/path/to/official_data/train_image_common')
    parser.add_argument('--common_label_json_file', type=str,
                        default='official_data/train_label_common.json')

    parser.add_argument('--train_image_special_root_path', type=str,
                       default='/path/to/official_data/train_image_special')
    parser.add_argument('--special_label_json_file', type=str,
                       default='/path/to/official_data/train_label_special.json')

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

    char_list = list(char_set)
    char_list.sort()

    with open('dictionary/chars.txt', 'a', encoding='utf-8') as out_file:
        for char in char_list:
            out_file.write('{}\n'.format(char))

'''bash
python recognizer/tools/extract_train_data.py
'''

