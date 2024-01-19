import glob
from PIL import Image
import os
import torch
import numpy as np

class EndoVis2017(torch.utils.data.Dataset):
    def __init__(self, dir_main, split, transform = None, imsize=None, task = "binary"):
        super(EndoVis2017, self).__init__()
        self.root_dir = dir_main

        if split == 'Train':
          dataset_num = 8
        elif split == 'Test':
          dataset_num = 10

        self.img_files = []
        self.mask_files = []
        for i in range(1, dataset_num + 1):
          data_path = os.path.join(self.root_dir, split, 'instrument_dataset_{}'.format(i))
          self.img_files.extend(glob.glob(os.path.join(data_path, 'images', '*')))
          if task == "binary":
            self.mask_files.extend(glob.glob(os.path.join(data_path, 'binary_masks', '*')))
          if task == "multi":
            self.mask_files.extend(glob.glob(os.path.join(data_path, 'instruments_masks', '*')))


        self.transform = transform
        # self.mask_transform=mask_transform
        self.imsize = imsize
        self.task = task


    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        with open(mask_path, 'rb') as f:
            mask = Image.open(f)
            # print(np.unique(np.array(mask)))
            if self.task == "binary":
                mask = mask.convert('L')
                # print(np.unique(np.array(mask)))
                mask = mask.point(lambda x: 1 if x > 0 else 0)
                # mask = mask.convert('L')  # or mask = mask.convert('1')
            if self.task == "multi":
                # print(np.unique(np.array(mask)))
                normalized_mask_array = np.array(mask) / 32.
                mask = Image.fromarray((normalized_mask_array).astype(np.uint8))
                mask = mask.convert('L')
                # print(np.unique(np.array(mask)))
        if self.imsize is not None:
            img = img.resize((self.imsize, self.imsize), resample=Image.BILINEAR)
            mask = mask.resize((self.imsize, self.imsize), resample=Image.NEAREST)
        if self.transform is not None:
            # mat, mat_inv = self.getTransformMat(self.imsize, True)
            img_np = np.array(img).astype(np.uint8)
            mask_np = np.array(mask).astype(np.uint8)
            transformed = self.transform(image=img_np, mask=mask_np)

            # Access the transformed image and mask
            # trans_img = transformed["image"]
            trans_img = torch.from_numpy(transformed['image'].transpose(2, 0, 1)) / 255.0
            trans_mask = torch.from_numpy(transformed["mask"])
            return trans_img, trans_mask.long(), index
        else:
            return img, mask
    def __len__(self):
        return len(self.img_files)
    

class EndoVis2018(torch.utils.data.Dataset):
    def __init__(self, dir_main, split, transform = None, imsize=None):
        super(EndoVis2018, self).__init__()
        self.root_dir = dir_main

        if split == 'Train':
          dataset_num = 15
        elif split == 'Test':
          dataset_num = 4

        self.img_files = []
        self.mask_files = []
        for i in range(1, dataset_num + 1):
          data_path = os.path.join(self.root_dir, split, 'seq_{}'.format(i))
          self.img_files.extend(glob.glob(os.path.join(data_path, 'images', '*')))
          self.mask_files.extend(glob.glob(os.path.join(data_path, 'binary_masks', '*')))

        self.transform = transform
        # self.mask_transform=mask_transform
        self.imsize = imsize


    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        with open(mask_path, 'rb') as f:
            mask = Image.open(f)
            mask = mask.point(lambda x: 1 if x > 0 else 0)
            # mask = mask.convert('L')  # or mask = mask.convert('1')
        if self.imsize is not None:
            img = img.resize((self.imsize, self.imsize), resample=Image.BILINEAR)
            mask = mask.resize((self.imsize, self.imsize), resample=Image.NEAREST)
        if self.transform is not None:
            # mat, mat_inv = self.getTransformMat(self.imsize, True)
            img_np = np.array(img).astype(np.uint8)
            mask_np = np.array(mask).astype(np.uint8)
            transformed = self.transform(image=img_np, mask=mask_np)

            # Access the transformed image and mask
            # trans_img = transformed["image"]
            trans_img = torch.from_numpy(transformed['image'].transpose(2, 0, 1)) / 255.0
            trans_mask = torch.from_numpy(transformed["mask"])
            return trans_img, trans_mask.long(), index
        else:
            return img, mask
    def __len__(self):
        return len(self.img_files)

    

class Robomis(torch.utils.data.Dataset):
    def __init__(self, dir_main, split, transform = None, imsize=None):
        super(Robomis, self).__init__()
        self.transform = transform
        # self.mask_transform=mask_transform
        self.imsize = imsize
        self.img_files = glob.glob(os.path.join(dir_main,'images',split,'*.png'))
        self.mask_files = []
        for img_path in self.img_files:
             self.mask_files.append(os.path.join(dir_main, 'annotations', split, os.path.basename(img_path)))

    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        with open(mask_path, 'rb') as f:
            mask = Image.open(f)
            mask = mask.point(lambda x: 1 if x > 0 else 0, mode='1')
            # mask = mask.convert('L')  # or mask = mask.convert('1')
        if self.imsize is not None:
            img = img.resize((self.imsize, self.imsize), resample=Image.BILINEAR)
            mask = mask.resize((self.imsize, self.imsize), resample=Image.NEAREST)
        if self.transform is not None:
            # mat, mat_inv = self.getTransformMat(self.imsize, True)
            img_np = np.array(img).astype(np.uint8)
            mask_np = np.array(mask).astype(np.uint8)
            transformed = self.transform(image=img_np, mask=mask_np)

            # Access the transformed image and mask
            # trans_img = transformed["image"]
            trans_img = torch.from_numpy(transformed['image'].transpose(2, 0, 1)) / 255.0
            trans_mask = torch.from_numpy(transformed["mask"])
        else:
            trans_img = img
            trans_mask = mask
        return trans_img, trans_mask.long(), index
    
    def __len__(self):
        return len(self.img_files)
    



class Autolapro(torch.utils.data.Dataset):
    def __init__(self, dir_main, split, transform = None, imsize=None):
        super(Autolapro, self).__init__()
        self.root_dir = dir_main

        if split == 'Train':
          dataset_range = range(170)
        elif split == 'Validation':
          dataset_range = range(170, 227)
        elif split == 'Test':
          dataset_range = range(227, 300)

        self.img_files = []
        self.mask_files = []
        for i in range(1, dataset_num + 1):
          data_path = os.path.join(self.root_dir, split, 'seq_{}'.format(i))
          self.img_files.extend(glob.glob(os.path.join(data_path, 'images', '*')))
          self.mask_files.extend(glob.glob(os.path.join(data_path, 'binary_masks', '*')))

        self.transform = transform
        # self.mask_transform=mask_transform
        self.imsize = imsize


    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        with open(mask_path, 'rb') as f:
            mask = Image.open(f)
            mask = mask.point(lambda x: 1 if x > 0 else 0)
            # mask = mask.convert('L')  # or mask = mask.convert('1')
        if self.imsize is not None:
            img = img.resize((self.imsize, self.imsize), resample=Image.BILINEAR)
            mask = mask.resize((self.imsize, self.imsize), resample=Image.NEAREST)
        if self.transform is not None:
            # mat, mat_inv = self.getTransformMat(self.imsize, True)
            img_np = np.array(img).astype(np.uint8)
            mask_np = np.array(mask).astype(np.uint8)
            transformed = self.transform(image=img_np, mask=mask_np)

            # Access the transformed image and mask
            # trans_img = transformed["image"]
            trans_img = torch.from_numpy(transformed['image'].transpose(2, 0, 1)) / 255.0
            trans_mask = torch.from_numpy(transformed["mask"])
            return trans_img, trans_mask.long(), index
        else:
            return img, mask
    def __len__(self):
        return len(self.img_files)