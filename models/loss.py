import torch 
import torch.nn as nn
from .models_utils import VGG19
import torch.nn.functional as F

def adv_loss(self, discriminator_preds, is_real):
    '''
    Computes adversarial loss from nested list of fakes outputs from discriminator.
    '''
    target = torch.ones_like if is_real else torch.zeros_like

    adv_loss = 0.0
    for preds in discriminator_preds:
        pred = preds[-1]
        adv_loss += F.mse_loss(pred, target(pred))
    return adv_loss

def fm_loss(self, real_preds, fake_preds):
    '''
    Computes feature matching loss from nested lists of fake and real outputs from discriminator.
    '''
    fm_loss = 0.0
    for real_features, fake_features in zip(real_preds, fake_preds):
        for real_feature, fake_feature in zip(real_features, fake_features):
            fm_loss += F.l1_loss(real_feature.detach(), fake_feature)
    return fm_loss

class vgg_loss(nn.Module):
    '''
    Loss Class
    Implements composite loss for GauGAN
    Values:
        lambda1: weight for feature matching loss, a float
        lambda2: weight for vgg perceptual loss, a float
        device: 'cuda' or 'cpu' for hardware to use
        norm_weight_to_one: whether to normalize weights to (0, 1], a bool
    '''

    def __init__(self, device='cuda'):
        super().__init__()
        self.vgg = VGG19().to(device)
        self.vgg_weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]


    def forward(self, x_real, x_fake):
        '''
        Computes perceptual loss with VGG network from real and fake images.
        '''
        vgg_real = self.vgg(x_real)
        vgg_fake = self.vgg(x_fake)

        vgg_loss = 0.0
        for real, fake, weight in zip(vgg_real, vgg_fake, self.vgg_weights):
            vgg_loss += weight * F.l1_loss(real.detach(), fake)
        return vgg_loss


def gen_loss(fake_preds_for_g, real_preds_for_d, img_o_fake, img_o_real, n_discriminators, lambda1=10., lambda2=10., norm_weight_to_one=True):

    lambda0 = 1.0
    # Keep ratio of composite loss, but scale down max to 1.0
    scale = max(lambda0, lambda1, lambda2) if norm_weight_to_one else 1.0

    lambda0 = lambda0 / scale
    lambda1 = lambda1 / scale
    lambda2 = lambda2 / scale

    g_loss = (
        lambda0 * adv_loss(fake_preds_for_g, True) + \
        lambda1 * fm_loss(real_preds_for_d, fake_preds_for_g) / n_discriminators + \
        lambda2 * vgg_loss(img_o_fake, img_o_real)
    )

    return g_loss


def den_loss(fake_preds_for_d, real_preds_for_d):

    d_loss = 0.5 * (adv_loss(real_preds_for_d, True) + adv_loss(fake_preds_for_d, False))

    return d_loss

