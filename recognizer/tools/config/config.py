def get_lines():
    return len(open('recognizer/tools/dictionary/chars.txt', 'r',encoding='utf-8').readlines())


num_class = get_lines() + 1

train_test_split=0.95



char_path='recognizer/tools/dictionary/chars.txt'

train_image_path='official_data/train_image_common'
train_json='official_data/train_label_common.json'
train_image_special_root_path='official_data/train_image_special'
special_label_json_file'official_data/train_label_special.json'


