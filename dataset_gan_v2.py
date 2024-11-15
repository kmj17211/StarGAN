from os.path import join
import os
import random

from PIL import Image
from scipy.io import loadmat

import torch
from torchvision.transforms import ToTensor
from torchvision import transforms as T
from torch.utils.data import Dataset
from torch.utils import data
from torchvision.datasets import ImageFolder

class SAR_Dataset(Dataset):
    def __init__(self, data_type, transform = False, train = True):
        super(SAR_Dataset).__init__()

        if transform:
            self.transform = transform
        else:
            self.transform = ToTensor()

        path = './Data/SAR Data/SAMPLE'
        self.data_type = data_type
        self.train = train
        if data_type == 'VGG':

            if train:
                self.dir_t = 'gen_complex'
            else:
                self.dir_t = 'real/train'

            dir = 'png_images_pix2pix_60%/qpm'

            self.path2png = join(path, dir, self.dir_t)
            path2folder = self.path2png
            self.getitem = self._vgg_getitem

        else:

            if train:
                self.dir_t = 'train'
            else:
                self.dir_t = 'test'

            if data_type == 'Complex':
                dir_an = 'mat_files_pix2pix_60%'
                dir_mg = 'png_images_pix2pix_60%/qpm'

                self.path2png = join(path, dir_mg)
                self.path2mat = join(path, dir_an)

                self.getitem = self._complex_getitem

            elif data_type == 'QPM':
                dir_mg = 'png_images/qpm'
                self.path2png = join(path, dir_mg)
                self.getitem = self._png_getitem

            else:
                raise Exception('Data Type을 Complex나 QPM 둘 중 하나로 입력해')
        
        label = ['2s1', 'bmp2', 'btr70', 'm1', 'm2', 'm35', 'm60', 'm548', 't72', 'zsu23']
        self.real_label = label[:3]
        self.synth_label = label[3:]
        
        self.data_name = []
        self.data_label = []
        if self.train:
            for label in self.real_label:
                path2data = join(self.path2png, 'real', label)
                data_name = os.listdir(path2data)
                self.data_name.extend(data_name)

                self.data_label.extend([label] * len(data_name))
        
        for label in self.synth_label:
            path2data = join(self.path2png, 'synth', label)
            data_name = os.listdir(path2data)
            self.data_name.extend(data_name)

            self.data_label.extend([label] * len(data_name))

    def _png_getitem(self, index):
        
        # Label
        label = self.data_label[index]

        if label in self.real_label:
            data_path = join(self.path2png, 'real', label, self.data_name[index])
            img = self.transform(Image.open(data_path).convert('L'))
            cls = torch.tensor(1, dtype = torch.float32)
        elif label in self.synth_label:
            data_path = join(self.path2png, 'synth', label, self.data_name[index])
            img = self.transform(Image.open(data_path).convert('L'))
            cls = torch.tensor(0, dtype = torch.float32)

        if self.train:
            return img, cls
        else:
            return img, cls, label, self.data_name[index]
    
    def __getitem__(self, index):
        return self.getitem(index)
    
    def __len__(self):
        return len(self.data_name)

class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if (i+1) < 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128, 
               batch_size=16, dataset='CelebA', mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if dataset == 'CelebA':
        dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode)
    elif dataset == 'RaFD':
        dataset = ImageFolder(image_dir, transform)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader