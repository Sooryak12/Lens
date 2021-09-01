import pandas as pd
import numpy as np
import cv2

train = pd.read_csv('dataset_yolo/train.csv')
train['img_h'], train['img_w'] = np.nan, np.nan
train_image_path = '/Volumes/Samsung_T5/yolov5_detector/dataset_temp/imgs_train/'
for i in range(len(train)):
  print(i)
  hashmapuh = {}
  x = train['Image_Name'].iloc[i]
  print(x)
  if x not in hashmapuh.keys():
        img = cv2.imread(train_image_path + x)
        img_h, img_w, img_c = img.shape
        hashmapuh[x] = (img_h, img_w)
  else:
        img_h, img_w = hashmapuh[x]

  train['img_h'][i], train['img_w'][i] = img_h, img_w

train.to_csv('dataset_yolo/train_clean.csv',index=False)