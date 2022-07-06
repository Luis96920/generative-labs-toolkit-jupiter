import torch 
import torch.nn as nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import time
from models.loss import Loss

# Parse torch version for autocast
# ######################################################
version = torch.__version__
version = tuple(int(n) for n in version.split('.')[:-1])
has_autocast = version >= (1, 6)
# ######################################################

def print_device_name():
  cuda = torch.device('cuda:0')
  torch.cuda.set_device(0)
  return print('Exp is running in {} No.{}'.format(
      torch.cuda.get_device_name(torch.cuda.current_device()), torch.cuda.current_device()))

      
def show_tensor_images(image_tensor):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:1], nrow=1)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def train(dataloader, models, optimizers, schedulers, device, desc, epochs=100):
    encoder, generator, discriminator = models
    g_optimizer, d_optimizer = optimizers
    g_scheduler, d_scheduler = schedulers

    loss_fn = Loss(device=device)

    display_step = 100
    cur_step = 0
    mean_g_loss = 0.0
    mean_d_loss = 0.0

    for epoch in tqdm(range(epochs), desc=desc, leave=True):
        # Training epoch
        # time
        since_load = time.time()
        for (x_real, labels, insts, bounds) in tqdm(dataloader, desc=f'  inner loop for epoch {epoch}', leave=False):
            x_real = x_real.to(device)
            labels = labels.to(device)
            insts = insts.to(device)
            bounds = bounds.to(device)

            # time
            time_elapsed_load = time.time() - since_load
            since_training = time.time()

            # Enable autocast to FP16 tensors (new feature since torch==1.6.0)
            # If you're running older versions of torch, comment this out
            # and use NVIDIA apex for mixed/half precision training
            if has_autocast:
                with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                    g_loss, d_loss, x_fake = loss_fn(
                        x_real, labels, insts, bounds, encoder, generator, discriminator
                    )
            else:
                g_loss, d_loss, x_fake = loss_fn(
                    x_real, labels, insts, bounds, encoder, generator, discriminator
                )

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            mean_g_loss += g_loss.item() / display_step
            mean_d_loss += d_loss.item() / display_step

            if cur_step % display_step == 0 and cur_step > 0:
                print('Step {}: Generator loss: {:.5f}, Discriminator loss: {:.5f}'
                      .format(cur_step, mean_g_loss, mean_d_loss))
                show_tensor_images(x_fake.to(x_real.dtype))
                show_tensor_images(x_real)
                mean_g_loss = 0.0
                mean_d_loss = 0.0
            cur_step += 1

            # time 
            time_elapsed_training = time.time() - since_training
            since_load = time.time()
            print('Loading images complete in {:.0f}m {:.0f}s {:.0f}ms'.format(
            time_elapsed_load // 60, time_elapsed_load % 60, 60*time_elapsed_load % 60))
            print('Training complete in {:.0f}m {:.0f}s {:.0f}ms'.format(
            time_elapsed_training // 60, time_elapsed_training % 60, 60*time_elapsed_training % 60))

        g_scheduler.step()
        d_scheduler.step()
