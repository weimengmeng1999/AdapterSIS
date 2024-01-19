import os
import sys
import cv2
import json
import numpy as np



def get_one_sample(root_dir, image_file, image_path, save_dir, mask,
                   class_name):
    if '.jpg' in image_file:
        suffix = '.jpg'
    elif '.png' in image_file:
        suffix = '.png'
    mask_path = os.path.join(
        save_dir,
        image_file.replace(suffix, '') + '_{}.png'.format(class_name))
    cv2.imwrite(mask_path, mask)
    data = {
        'img_path': image_path.replace(root_dir, ''),
        'mask_path': mask_path.replace(root_dir, ''),
    }
    return data


def process(root_dir, data_file):
    data_list = []
    data_type = os.getenv('DATA_TYPE', 'train')
    if data_type == 'train':
        image_dir = os.path.join(root_dir, 'Training')
    elif data_type == 'test':
        image_dir = os.path.join(root_dir, 'Testing')

    for image_file in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_file)
        if 'img' in image_file:
            print(image_path)
            image = cv2.imread(image_path)
            mask = cv2.imread(image_path.replace(
                '_img.png', '_label.png'))
            mask = mask[:, :, 0]

            for class_id, class_name in enumerate(['background',
                                                   'instrument']):
                if class_name == 'background':
                    target_mask = (mask == 0) * 255
                elif class_name == 'instrument':
                    target_mask = (mask > 0) * 255



if __name__ == '__main__':
    root_dir = sys.argv[1]
    data_file = sys.argv[2]
    process(root_dir, data_file)