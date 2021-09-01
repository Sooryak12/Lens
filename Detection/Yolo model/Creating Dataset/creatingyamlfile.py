from os.path import join
from glob import glob
import yaml

cwd = '/Volumes/Samsung_T5/yolov5_detector/dataset_yolo'

classes = ['text']

with open(join( cwd , 'train.txt'), 'w') as f:
    for path in glob('/Volumes/Samsung_T5/yolov5_detector/dataset_yolo/images/train/*'):
        f.write(path+'\n')

with open(join( cwd , 'val.txt'), 'w') as f:
    for path in glob('/Volumes/Samsung_T5/yolov5_detector/dataset_yolo/images/valid/*'):
        f.write(path+'\n')

data = dict(
    train =  join( cwd , 'train.txt') ,
    val   =  join( cwd , 'val.txt' ),
    nc    = 1,
    names = classes
    )

with open(join( cwd , 'text.yaml'), 'w') as outfile:
    yaml.dump(data, outfile, default_flow_style=False)

f = open(join( cwd , 'text.yaml'), 'r')
print('\nyaml:')
print(f.read())