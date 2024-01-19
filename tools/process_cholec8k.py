import os
import sys
import cv2
import json
import numpy as np

class_list = ['black_background', 'abdominal_wall', 'liver',
              'gastrointestinal_tract', 'fat', 'grasper',
              'connective_tissue', 'blood', 'cystic_duct',
              'l_hook_electrocautery', 'gallbladder', 'hepatic_vein',
              'liver_ligament']

# All
video_dir_list = ['video01', 'video09', 'video12', 'video17', 'video18', 'video20',
                  'video24', 'video25', 'video26', 'video27', 'video28', 'video35',
                  'video37', 'video43', 'video48', 'video52', 'video55']

# A spatio-temporal network for video semantic segmentation in surgical videos
# train_video_dir_list = ['video01', 'video09', 'video17', 'video18', 'video24',
#                         'video25', 'video26', 'video27', 'video28', 'video35',
#                         'video37', 'video43', 'video52']
# val_video_dir_list = []
# test_video_dir_list = ['video12', 'video20', 'video48', 'video55']

# Class-wise confidence-aware active learning for laparoscopic images segmentation
train_video_dir_list = ['video01', 'video09', 'video17', 'video18', 'video20',
                        'video24', 'video25', 'video26', 'video27', 'video28', 'video35',
                        'video37', 'video43']
val_video_dir_list = []
test_video_dir_list = ['video12', 'video48', 'video52', 'video55']




class2rgb = {
    'black_background': (50, 50, 50),
    'abdominal_wall': (11, 11, 11),
    'liver': (21, 21, 21),
    'gastrointestinal_tract': (13, 13, 13),
    'fat': (12, 12, 12),
    'grasper': (31, 31, 31),
    'connective_tissue': (23, 23, 23),
    'blood': (24, 24, 24),
    'cystic_duct': (25, 25, 25),
    'l_hook_electrocautery': (32, 32, 32),
    'gallbladder': (22, 22, 22),
    'hepatic_vein': (33, 33, 33),
    'liver_ligament': (5, 5, 5),
}


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

    return data


def process(root_dir, data_file):
    data_list = []
    data_type = os.getenv('DATA_TYPE', 'train')
    if data_type == 'train':
        video_dir_list = train_video_dir_list
    elif data_type == 'val':
        video_dir_list = val_video_dir_list
    elif data_type == 'test':
        video_dir_list = test_video_dir_list
    for video_dir in video_dir_list:
        for image_dir in os.listdir(os.path.join(root_dir, video_dir)):
            print(os.path.join(root_dir, video_dir, image_dir))

            for image_file in os.listdir(os.path.join(root_dir, video_dir, image_dir)):
                image_path = os.path.join(
                    root_dir, video_dir, image_dir, image_file)
                if 'mask' not in image_file:
                    image = cv2.imread(image_path)
                    mask = cv2.imread(image_path.replace(
                        '.png', '_watershed_mask.png'))
                    mask = mask[:, :, 0]

                    for class_id, class_name in enumerate(class_list):
                        target_mask = (mask == class2rgb[class_name][0]) * 255


if __name__ == '__main__':
    root_dir = sys.argv[1]
    data_file = sys.argv[2]
    process(root_dir, data_file)