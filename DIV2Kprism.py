import torch.utils.data as data
import os.path
import cv2
import numpy as np
import common
from PIL import Image

def default_loader(path):
    data=cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return data[:, :, [2, 1, 0]]
    #data = Image.open(path).convert('RGB')
    #return data

def npy_loader(path):
    return np.load(path)

IMG_EXTENSIONS = [
    '.png', '.npy',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

class CNNdiv2k(data.Dataset):
    def __init__(self, data_dir, ext):
        self.data_dir = data_dir
        self.scale = 4
        self.n_train = 31
        self.patch_size = 96
        #self.root = self.opt.root
        self.ext = ext #'.npy' # self.opt.ext   # '.png' or '.npy'(default)
        self.train = True #if self.opt.phase == 'train' else False
        self.repeat = 10#self.opt.test_every // (self.opt.n_train // self.opt.batch_size)
        self._set_filesystem()
        self.images_hr, self.images_lr = self._scan()

    def _set_filesystem(self): #, dir_data):
        #self.dir_hr = 'SRCT/project_data/train_npy_files/high_resolution'
        #self.dir_lr = 'SRCT/project_data/train_npy_files/low_resolution'
        self.dir_hr = self.data_dir + '/high_resolution' #'SRCT/project_data/train_npy_files/high_resolution'
        self.dir_lr = self.data_dir + '/low_resolution' #'SRCT/project_data/train_npy_files/low_resolution'

    def __getitem__(self, idx):
        lr, hr = self._load_file(idx)
        lr, hr = self._get_patch(lr, hr)
        lr, hr = common.set_channel(lr, hr, n_channels=1) #self.opt.n_colors)
        lr_tensor, hr_tensor = common.np2Tensor(lr, hr, rgb_range=1) #self.opt.rgb_range)
        return lr_tensor, hr_tensor

    def __len__(self):
        if self.train:
            return self.n_train*self.repeat #self.opt.n_train * self.repeat

    def _get_index(self, idx):
        if self.train:
            return idx % self.n_train
        else:
            return idx

    def _get_patch(self, img_in, img_tar):
        patch_size = self.patch_size
        scale = self.scale
        if self.train:
            img_in, img_tar = common.get_patch(
                img_in, img_tar, patch_size=patch_size, scale=scale)
            img_in, img_tar = common.augment(img_in, img_tar)
        else:
            ih, iw = img_in.shape[:2]
            img_tar = img_tar[0:ih * scale, 0:iw * scale, :]
        return img_in, img_tar

    def _scan(self):
        list_hr = sorted(make_dataset(self.dir_hr))
        list_lr = sorted(make_dataset(self.dir_lr))
        return list_hr, list_lr

    def _load_file(self, idx):
        idx = self._get_index(idx)
        if self.ext == '.npy':
            lr = npy_loader(self.images_lr[idx])
            hr = npy_loader(self.images_hr[idx])
        else:
            lr = default_loader(self.images_lr[idx])
            hr = default_loader(self.images_hr[idx])
        return lr, hr

class not_used_div2k(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.scale = self.opt.scale
        self.root = self.opt.root
        self.ext = self.opt.ext   # '.png' or '.npy'(default)
        self.train = True if self.opt.phase == 'train' else False
        self.repeat = 10#self.opt.test_every // (self.opt.n_train // self.opt.batch_size)
        self._set_filesystem(self.root)
        self.images_hr, self.images_lr = self._scan()

    def _set_filesystem(self, dir_data):
        self.root = dir_data
        self.dir_hr = '/home/fried/git/jason/esrt2/train_npy_files/high_resolution'
        self.dir_lr = '/home/fried/git/jason/esrt2/train_npy_files/low_resolution'

    def __getitem__(self, idx):
        lr, hr = self._load_file(idx)
        lr, hr = self._get_patch(lr, hr)
        lr, hr = common.set_channel(lr, hr, n_channels=self.opt.n_colors)
        lr_tensor, hr_tensor = common.np2Tensor(lr, hr, rgb_range=self.opt.rgb_range)
        return lr_tensor, hr_tensor

    def __len__(self):
        if self.train:
            return self.opt.n_train * self.repeat

    def _get_index(self, idx):
        if self.train:
            return idx % self.opt.n_train
        else:
            return idx

    def _get_patch(self, img_in, img_tar):
        patch_size = self.opt.patch_size
        scale = self.scale
        if self.train:
            img_in, img_tar = common.get_patch(
                img_in, img_tar, patch_size=patch_size, scale=scale)
            img_in, img_tar = common.augment(img_in, img_tar)
        else:
            ih, iw = img_in.shape[:2]
            img_tar = img_tar[0:ih * scale, 0:iw * scale, :]
        return img_in, img_tar

    def _scan(self):
        list_hr = sorted(make_dataset(self.dir_hr))
        list_lr = sorted(make_dataset(self.dir_lr))
        return list_hr, list_lr

    def _load_file(self, idx):
        idx = self._get_index(idx)
        if self.ext == '.npy':
            lr = npy_loader(self.images_lr[idx])
            hr = npy_loader(self.images_hr[idx])
        else:
            lr = default_loader(self.images_lr[idx])
            hr = default_loader(self.images_hr[idx])
        return lr, hr
