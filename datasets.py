#import h5py
import numpy as np
from torch.utils.data import Dataset

class CNNdiv2k(Dataset):
    def __init__(self):
        #self.opt = opt
        self.scale = 4
        self.n_train = 31
        self.patch_size = 96
        #self.root = self.opt.root
        self.ext = '.npy' # self.opt.ext   # '.png' or '.npy'(default)
        self.train = True #if self.opt.phase == 'train' else False
        self.repeat = 10#self.opt.test_every // (self.opt.n_train // self.opt.batch_size)
        self._set_filesystem()
        self.images_hr, self.images_lr = self._scan()

    def _set_filesystem(self): #, dir_data):
        #self.root = dir_data
        self.dir_hr = '/home/fried/git/jason/esrt2/npy files/train_npy'
        self.dir_lr = '/home/fried/git/jason/esrt2/npy files/trainx4_npy'

    def __getitem__(self, idx):
        lr, hr = self._load_file(idx)
        lr, hr = self._get_patch(lr, hr)
        lr, hr = common.set_channel(lr, hr) #, n_channels=self.opt.n_colors)
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
class TrainDataset(Dataset):
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][idx] / 255., 0), np.expand_dims(f['hr'][idx] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


class EvalDataset(Dataset):
    def __init__(self, h5_file):
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][str(idx)][:, :] / 255., 0), np.expand_dims(f['hr'][str(idx)][:, :] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])
