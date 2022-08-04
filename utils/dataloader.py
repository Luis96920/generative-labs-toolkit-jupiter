import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
import os
import numpy as np
from utils.utils import is_distributed


def create_loaders(train_dir, target_width, batch_size, n_classes, world_size, rank):
    
    dataset = SwordSorceryDataset(train_dir, target_width=target_width, n_classes=n_classes)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=0, drop_last=False) if is_distributed() else None
    print('Sampler:')
    print(sampler)

    loader = DataLoader(dataset, batch_size=batch_size,
                                  collate_fn=SwordSorceryDataset.collate_fn,
                                  num_workers=1, pin_memory=False, 
                                  shuffle=(sampler is None),
                                  sampler=sampler)
   
    return loader 


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

    def __init__(self, paths, target_width=1024, n_classes=2):
        super().__init__()

        self.n_classes = n_classes
        self.paths = paths
        self.paths['path_inputs'] = {key: val for key, val in self.paths['path_inputs'].items() if os.path.isdir(os.path.join(self.paths['path_root'], val))}
        self.n_inputs = len(self.paths['path_inputs'])


        # Collect list of examples
        self.examples = {}
        self.load_examples_from_dir()
        self.examples = list(self.examples.values())
        self.examples=[example for example in self.examples if len(example)==self.n_inputs]
        assert all(len(example) == self.n_inputs for example in self.examples)

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
         

    def load_examples_from_dir(self):
        '''
        Given a folder of examples, this function returns a list of paired examples.
        '''
        for attr, path_input  in self.paths['path_inputs'].items():
            abs_path = os.path.join(self.paths['path_root'], path_input)
            assert os.path.isdir(abs_path)
            for root, _, files in os.walk(abs_path):
                for f in files:
                    if f.split('.')[1] in ('jpg', 'jpeg', 'png'):
                        file_name = f.split('.')[0] # keep name, remove .jpg or .png

                        if file_name not in self.examples.keys():
                            self.examples[file_name] = {}
                        self.examples[file_name][attr] = root + '/' + f

        """
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
        """
    
    def __getitem__(self, idx):
        example = self.examples[idx]

        # Load input and output images
        img_i = Image.open(example['input_img']).convert('RGB')  # color image: (3, 512, 1024)
        img_o = Image.open(example['output_img']).convert('RGB')  # color image: (3, 512, 1024)

        # Apply corresponding transforms
        img_i = self.img_transforms(img_i)
        img_o = self.img_transforms(img_o)

        # Load optional inst images
        if 'inst_map' in self.paths['path_inputs'].keys():
            inst = Image.open(example['inst_map']).convert('L')   # instance map: (512, 1024)
            inst = self.map_transforms(inst)

            # Convert instance map to instance boundary map
            bound = torch.ByteTensor(inst.shape).zero_()
            bound[:, :, 1:] = bound[:, :, 1:] | (inst[:, :, 1:] != inst[:, :, :-1])
            bound[:, :, :-1] = bound[:, :, :-1] | (inst[:, :, 1:] != inst[:, :, :-1])
            bound[:, 1:, :] = bound[:, 1:, :] | (inst[:, 1:, :] != inst[:, :-1, :])
            bound[:, :-1, :] = bound[:, :-1, :] | (inst[:, 1:, :] != inst[:, :-1, :])
            bound = bound.to(img_i.dtype)
        else:   
            inst = torch.zeros(0, img_i.shape[1], img_i.shape[2])
            bound = torch.zeros(0, img_i.shape[1], img_i.shape[2])

        # Load optional label images
        if 'label_map' in self.paths['path_inputs'].keys():
            label = Image.open(example['label_map']).convert('L') # semantic label map: (512, 1024)
            label = self.map_transforms(label)#.long() * 255 # --> check this!

            # Convert labels to one-hot vectors
            #label = torch.zeros(self.n_classes, img.shape[1], img.shape[2]).scatter_(0, label, 1.0).to(img.dtype)
        else:   
            label = torch.zeros(0, img_i.shape[1], img_i.shape[2])
 
        return (img_i, label, inst, bound, img_o)

    def __len__(self):
        return len(self.examples)

    @staticmethod
    def collate_fn(batch):
        imgs_i, labels, insts, bounds, imgs_o = [], [], [], [], []
        for (x, l, i, b, o) in batch:
            imgs_i.append(x)
            labels.append(l)
            insts.append(i)
            bounds.append(b)
            imgs_o.append(o)
        return (
            torch.stack(imgs_i, dim=0),
            torch.stack(labels, dim=0),
            torch.stack(insts, dim=0),
            torch.stack(bounds, dim=0),
            torch.stack(imgs_o, dim=0),

        )

    def get_input_size_g(self):
        img_i, label, inst, bound, img_o = self.__getitem__(1)
        return img_i.shape[0] + label.shape[0] + bound.shape[0]

    def get_input_size_d(self):
        img_i, label, inst, bound, img_o = self.__getitem__(1)
        return bound.shape[0] + label.shape[0] + img_o.shape[0]
 
    # it's remaining input_size_encoder because we're not using that for now
