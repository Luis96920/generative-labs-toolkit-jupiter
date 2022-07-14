import torch 
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


def print_device_name():
  cuda = torch.device('cuda:0')
  torch.cuda.set_device(0)
  return print('Exp is running in {} No.{}'.format(
      torch.cuda.get_device_name(torch.cuda.current_device()), torch.cuda.current_device()))


def show_tensor_images(image_tensor):
    '''
    Function for visualizing images: Given a tensor of imagess, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:1], nrow=1)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()