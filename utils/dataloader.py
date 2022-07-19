import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np


def scale_width(img, target_width, method):
    '''
    Function that scales an image to target_width while retaining aspect ratio.
    '''
    w, h = img.size
    if w == target_width: return img
    target_height = target_width * h // w
    return img.resize((target_width, target_height), method)

class SwordSorceryDataset(torch.utils.data.Dataset):
    '''
    SwordSorceryDataset Class
    Values:
        paths: (a list of) paths to load examples from, a list or string
        target_width: the size of image widths for resizing, a scalar
        n_classes: the number of object classes, a scalar
    '''

    def __init__(self, paths, target_width=1024, n_classes=2, n_inputs=2):
        super().__init__()

        self.n_classes = n_classes

        # Collect list of examples
        self.examples = {}
        if type(paths) == str:
            self.load_examples_from_dir(paths)
        elif type(paths) == list:
            for path in paths:
                self.load_examples_from_dir(path)
        else:
            raise ValueError('`paths` should be a single path or list of paths')

        self.examples = list(self.examples.values())
        self.examples=[example for example in self.examples if len(example)==n_inputs]
        assert all(len(example) == n_inputs for example in self.examples)

        # Initialize transforms for the real color image
        self.img_transforms = transforms.Compose([
            transforms.Lambda(lambda img: scale_width(img, target_width, Image.BICUBIC)),
            transforms.Lambda(lambda img: np.array(img)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        # Initialize transforms for semantic label and instance maps
        self.map_transforms = transforms.Compose([
            transforms.Lambda(lambda img: scale_width(img, target_width, Image.NEAREST)),
            transforms.Lambda(lambda img: np.array(img)),
            transforms.ToTensor(),
        ])

    def load_examples_from_dir(self, paths):
        '''
        Given a folder of examples, this function returns a list of paired examples.
        '''
        for path_input, attr in paths['path_inputs']:
          abs_path = os.path.join(paths['path_root'], path_input)
          assert os.path.isdir(abs_path)
          for root, _, files in os.walk(abs_path):
            for f in files:
              if f.split('.')[1] in ('jpg', 'jpeg', 'png'):
                file_name = f.split('.')[0] # keep name, remove .jpg or .png

                if file_name not in self.examples.keys():
                    self.examples[file_name] = {}
                self.examples[file_name][attr] = root + '/' + f

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Load image and maps
        img = Image.open(example['orig_img']).convert('RGB')  # color image: (3, 512, 1024)
        inst = Image.open(example['inst_map']).convert('L')   # instance map: (512, 1024)
        label = Image.open(example['label_map']).convert('L') # semantic label map: (512, 1024)

        # Apply corresponding transforms
        img = self.img_transforms(img)
        inst = self.map_transforms(inst)
        label = self.map_transforms(label).long() * 255

        # Convert labels to one-hot vectors
        #label = torch.zeros(self.n_classes, img.shape[1], img.shape[2]).scatter_(0, label, 1.0).to(img.dtype)

        # Convert instance map to instance boundary map
        bound = torch.ByteTensor(inst.shape).zero_()
        bound[:, :, 1:] = bound[:, :, 1:] | (inst[:, :, 1:] != inst[:, :, :-1])
        bound[:, :, :-1] = bound[:, :, :-1] | (inst[:, :, 1:] != inst[:, :, :-1])
        bound[:, 1:, :] = bound[:, 1:, :] | (inst[:, 1:, :] != inst[:, :-1, :])
        bound[:, :-1, :] = bound[:, :-1, :] | (inst[:, 1:, :] != inst[:, :-1, :])
        bound = bound.to(img.dtype)

        return (img, label, inst, bound)

    def __len__(self):
        return len(self.examples)

    @staticmethod
    def collate_fn(batch):
        imgs, labels, insts, bounds = [], [], [], []
        for (x, l, i, b) in batch:
            imgs.append(x)
            labels.append(l)
            insts.append(i)
            bounds.append(b)
        return (
            torch.stack(imgs, dim=0),
            torch.stack(labels, dim=0),
            torch.stack(insts, dim=0),
            torch.stack(bounds, dim=0),
        )