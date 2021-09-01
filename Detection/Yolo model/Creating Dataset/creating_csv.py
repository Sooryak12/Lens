import json
from glob import glob
import csv

def create_csv(path,flag,first_time,images_type):
    with open(path, 'r') as jsonfile:
        with open('dataset_yolo/train.csv', flag) as csvfile:
            dict_ = json.load(jsonfile)
            csvwriter = csv.writer(csvfile)
            if first_time:
                csvwriter.writerow(['Image_Name','Points','Label'])
            for img in images_type:
                    img_name = img
                    for text in dict_[img]:
                            points = text['points']
                            label = text['label']
                            nrow = len(points)
                            ncol = len(points[0])
                            if nrow != 4:
                                continue
                            if ncol != 2:
                                continue
                            try:
                                ele1 = points[0][0]
                                ele2 = points[0][1]
                                ele3 = points[1][0]
                                ele4 = points[1][1]
                                ele5 = points[2][0]
                                ele6 = points[2][1]
                                ele7 = points[3][0]
                                ele8 = points[3][1]
                                points = [ele1, ele2, ele3, ele4, ele5, ele6, ele7, ele8]
                            except:
                                continue
                            csvwriter.writerow([img_name,points,label])

if __name__ == '__main__':

    images_common = [f.split('/')[-1] for f in glob('dataset/train_image_common/*')]
    images_special = [f.split('/')[-1] for f in glob('dataset/train_image_special/*')]
    test_images = [f.split('/')[-1] for f in glob('dataset/test_image/*')]

    path_to_json_common = 'dataset/train_label_common.json'
    path_to_json_special = 'dataset/train_label_special.json'

    create_csv(path_to_json_common,'w',True,images_common)
    create_csv(path_to_json_special,'a',False,images_special)
