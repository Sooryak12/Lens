import pandas as pd

train = pd.read_csv('/Volumes/Samsung_T5/yolov5_detector/dataset_yolo/train_bbox.csv')

unique_names = train.Image_Name.unique()
path = '/Volumes/Samsung_T5/yolov5_detector/dataset_temp/label_train/'

for name in unique_names:
    with open(path + name.split('.')[0]+'.txt','w') as f:
        for bbox in train[train.Image_Name == name].bounding_box:
            bbox = bbox.split(',')
            bounding_box = f'{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]}'
            print(bounding_box)
            f.write(bounding_box+'\n')