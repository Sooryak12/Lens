import pandas as pd
import shutil
import os
from tqdm import tqdm
from glob import glob
from sklearn import model_selection

df = pd.read_csv('/Volumes/Samsung_T5/yolov5_detector/dataset_yolo/train_bbox.csv')
path = '/Volumes/Samsung_T5/yolov5_detector/dataset_temp/imgs_train/'

unique_names = df.Image_Name.unique()
unique_names_train, unique_names_valid = model_selection.train_test_split(unique_names, random_state=42, shuffle=True,test_size=0.1)

train_files = [path + name for name in unique_names_train]
val_files   = [path + name for name in unique_names_valid]
# test_files = glob('/Volumes/Samsung_T5/yolov5_detector/dataset/test_image/*')

label_dir = '/Volumes/Samsung_T5/yolov5_detector/dataset_temp/label_train'

# os.makedirs('/Volumes/Samsung_T5/yolov5_detector/dataset_yolo/images/train', exist_ok = True)
# os.makedirs('/Volumes/Samsung_T5/yolov5_detector/dataset_yolo/images/valid', exist_ok = True)
# os.makedirs('/Volumes/Samsung_T5/yolov5_detector/dataset_yolo/labels/train', exist_ok = True)
# os.makedirs('/Volumes/Samsung_T5/yolov5_detector/dataset_yolo/labels/valid', exist_ok = True)

for file in tqdm(train_files):
    # shutil.copy(file, '/Volumes/Samsung_T5/yolov5_detector/dataset_yolo/images/train')
    filename = file.split('/')[-1].split('.')[0]
    shutil.copy(os.path.join(label_dir, filename+'.txt'), '/Volumes/Samsung_T5/yolov5_detector/dataset_yolo/labels/train')

for file in tqdm(val_files):
    # shutil.copy(file, '/Volumes/Samsung_T5/yolov5_detector/dataset_yolo/images/valid')
    filename = file.split('/')[-1].split('.')[0]
    shutil.copy(os.path.join(label_dir, filename+'.txt'), '/Volumes/Samsung_T5/yolov5_detector/dataset_yolo/labels/valid')


# for file in tqdm(test_files):
#     shutil.copy(file, '/Volumes/Samsung_T5/yolov5_detector/dataset_yolo/test')