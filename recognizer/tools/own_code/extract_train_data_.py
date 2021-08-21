import os 
import argparse
from skimage import io

def extract_train_data(image_path,json_path):
    with open(json_path) as infile:
        f =json.load(infile)
        for image_name ,box_with_labels in  f.items():
            img=io.imread(os.path.join(image_path,image_name))
            
            
        