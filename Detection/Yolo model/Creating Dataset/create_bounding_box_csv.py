import pandas as pd
import numpy as np
import ast

def get_xc_yc_h_w(point,img_h,img_w):
    point = ast.literal_eval(point)
    x1 = point[0]
    y1 = point[1]
    x2 = point[2]
    y2 = point[5]
    x_c = (x2 + x1)/2
    y_c = (y2 + y1)/2
    h = y2-y1
    w = x2-x1
    x_c /= img_w
    w /= img_w
    y_c /= img_h
    h /= img_h
    return f"{0}, {x_c}, {y_c}, {w}, {h}"

if __name__ == '__main__':
    df = pd.read_csv('/Volumes/Samsung_T5/yolov5_detector/dataset_yolo/train_clean.csv')
    df['bounding_box'] = -1
    for i in range(len(df)):
        print(i)
        df['bounding_box'].iloc[i] = get_xc_yc_h_w(df['Points'].iloc[i], df['img_h'].iloc[i], df['img_w'].iloc[i])

    df.to_csv('dataset_yolo/train_bbox.csv',index=False)



