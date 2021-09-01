from glob import glob
import cv2

def transfer_images(image_names,images_path):
    for img_name in image_names:
        img_bgr = cv2.imread(images_path+img_name)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        cpypath = 'dataset_temp/imgs_train/{}'.format(img_name)
        cv2.imwrite(cpypath, img_rgb)

if __name__ == '__main__':
    images_common = [f.split('/')[-1] for f in glob('dataset/train_image_common/*')]
    images_special = [f.split('/')[-1] for f in glob('dataset/train_image_special/*')]
    test_images = [f.split('/')[-1] for f in glob('dataset/test_image/*')]

    transfer_images(images_common, 'dataset/train_image_common/')
    transfer_images(images_special, 'dataset/train_image_special/')
    transfer_images(test_images, 'dataset/test_image/')