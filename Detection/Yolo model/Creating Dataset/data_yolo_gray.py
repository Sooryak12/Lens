import os
import shutil
from glob import glob
import cv2

# def write_to_gray(image_paths, cpy_path):
#     for image_path in image_paths:
#         img = cv2.imread(image_path)
#         img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         cv2.imwrite(cpy_path + image_path.split('/')[-1],img_gray)

# image_paths = glob('/Volumes/Samsung_T5/yolov5_detector/dataset_yolo/images/train/*')
# cpy_path = '/Volumes/Samsung_T5/yolov5_detector/dataset_yolo_gray/images/train/'
# write_to_gray(image_paths,  cpy_path)

# image_paths = glob('/Volumes/Samsung_T5/yolov5_detector/dataset_yolo/images/valid/*')
# cpy_path = '/Volumes/Samsung_T5/yolov5_detector/dataset_yolo_gray/images/valid/'
# write_to_gray(image_paths,  cpy_path)

image_paths = glob('/Volumes/Samsung_T5/yolov5_detector/dataset_yolo/test/*')
cpy_path = '/Volumes/Samsung_T5/yolov5_detector/dataset_yolo_gray/test/'
for image_path in image_paths:
    shutil.copy(image_path,  cpy_path+image_path.split('/')[-1])

# source_labels = glob("/Volumes/Samsung_T5/yolov5_detector/dataset_yolo/labels/train/*")
 
# # Destination path
# destination = "/Volumes/Samsung_T5/yolov5_detector/dataset_yolo_gray/labels/train/"

# for source_label in source_labels:
#    shutil.copy(source_label, destination+source_label.split('/')[-1])
# # print("File copied successfully.")

# source_labels = glob("/Volumes/Samsung_T5/yolov5_detector/dataset_yolo/labels/valid/*")
 
# # Destination path
# destination = "/Volumes/Samsung_T5/yolov5_detector/dataset_yolo_gray/labels/valid/"

# for source_label in source_labels:
#    shutil.copy(source_label, destination+source_label.split('/')[-1])

# print(len(glob('/Volumes/Samsung_T5/yolov5_detector/dataset_yolo_gray/images/train/*')))
# print(len(glob('/Volumes/Samsung_T5/yolov5_detector/dataset_yolo_gray/images/valid/*')))
# print(len(glob('/Volumes/Samsung_T5/yolov5_detector/dataset_yolo_gray/labels/train/*')))
# print(len(glob('/Volumes/Samsung_T5/yolov5_detector/dataset_yolo_gray/labels/valid/*')))
print(len(glob('/Volumes/Samsung_T5/yolov5_detector/dataset_yolo_gray/test/*')))




