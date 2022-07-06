import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.discriminators import MultiscaleDiscriminator
from models.generators import GlobalGenerator, LocalEnhancer
from models.loss import Loss
from models.models_utils import Encoder
from utils.dataloader import SwordSorceryDataset
from utils.utils import print_device_name, train



# init params
n_classes = 2                  # total number of object classes
rgb_channels = n_features = 3
device = 'cuda'
train_dir = [{
    'path_root': '/content/drive/MyDrive/GenerativeLabs/dataset/sword_sorcery_data_for_ramiro/paired_data',
    'path_inputs': [('02_output', 'orig_img'), ('01_segmented_input', 'inst_map'), ('01_segmented_input', 'label_map')]
    }]
epochs = 20                    # total number of train epochs
decay_after = 500              # number of epochs with constant lr
lr = 0.0005
betas = (0.5, 0.999)

# functions
def lr_lambda(epoch, epochs=200, decay_after=200):
    ''' Function for scheduling learning '''
    return 1. if epoch < decay_after else 1 - float(epoch - decay_after) / (epochs - decay_after)


def weights_init(m):
    ''' Function for initializing all model weights '''
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0., 0.02)

### Init train
## Phase 1: Low Resolution (1024 x 512)
dataloader1 = DataLoader(
    SwordSorceryDataset(train_dir, target_width=1024, n_classes=n_classes, n_inputs=3),
    collate_fn=SwordSorceryDataset.collate_fn, batch_size=2, shuffle=True, drop_last=False, pin_memory=True,
)
encoder = Encoder(rgb_channels, n_features).to(device).apply(weights_init)
#generator1 = GlobalGenerator(n_classes + n_features + 1, rgb_channels).to(device).apply(weights_init)
#discriminator1 = MultiscaleDiscriminator(n_classes + 1 + rgb_channels, n_discriminators=2).to(device).apply(weights_init)
#HARCODED BECAUSE LABEL IMAGE!  n_classes=1
generator1 = GlobalGenerator(1 + n_features + 1, rgb_channels).to(device).apply(weights_init)
discriminator1 = MultiscaleDiscriminator(1 + 1 + rgb_channels, n_discriminators=2).to(device).apply(weights_init)

g1_optimizer = torch.optim.Adam(list(generator1.parameters()) + list(encoder.parameters()), lr=lr, betas=betas)
d1_optimizer = torch.optim.Adam(list(discriminator1.parameters()), lr=lr, betas=betas)
g1_scheduler = torch.optim.lr_scheduler.LambdaLR(g1_optimizer, lr_lambda)
d1_scheduler = torch.optim.lr_scheduler.LambdaLR(d1_optimizer, lr_lambda)


## Phase 2: High Resolution (2048 x 1024)
dataloader2 = DataLoader(
    SwordSorceryDataset(train_dir, target_width=2048, n_classes=n_classes, n_inputs=3),
    collate_fn=SwordSorceryDataset.collate_fn, batch_size=1, shuffle=True, drop_last=False, pin_memory=True,
)
#generator2 = LocalEnhancer(n_classes + n_features + 1, rgb_channels).to(device).apply(weights_init)
#discriminator2 = MultiscaleDiscriminator(n_classes + 1 + rgb_channels).to(device).apply(weights_init)
#HARCODED BECAUSE LABEL IMAGE!  n_classes=1
generator2 = LocalEnhancer(1 + n_features + 1, rgb_channels).to(device).apply(weights_init)
discriminator2 = MultiscaleDiscriminator(1 + 1 + rgb_channels).to(device).apply(weights_init)

g2_optimizer = torch.optim.Adam(list(generator2.parameters()) + list(encoder.parameters()), lr=lr, betas=betas)
d2_optimizer = torch.optim.Adam(list(discriminator2.parameters()), lr=lr, betas=betas)
g2_scheduler = torch.optim.lr_scheduler.LambdaLR(g2_optimizer, lr_lambda)
d2_scheduler = torch.optim.lr_scheduler.LambdaLR(d2_optimizer, lr_lambda)

### Training
print_device_name()

# Phase 1: Low Resolution
#######################################################################
train(
    dataloader1,
    [encoder, generator1, discriminator1],
    [g1_optimizer, d1_optimizer],
    [g1_scheduler, d1_scheduler],
    device,
    desc='Epoch loop G1',
)


# Phase 2: High Resolution
#######################################################################
# Update global generator in local enhancer with trained
generator2.g1 = generator1.g1

# Freeze encoder and wrap to support high-resolution inputs/outputs
def freeze(encoder):
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    @torch.jit.script
    def forward(x, inst):
        x = F.interpolate(x, scale_factor=0.5, recompute_scale_factor=True)
        inst = F.interpolate(inst.float(), scale_factor=0.5, recompute_scale_factor=True)
        feat = encoder(x, inst.int())
        return F.interpolate(feat, scale_factor=2.0, recompute_scale_factor=True)
    return forward

train(
    dataloader2,
    [freeze(encoder), generator2, discriminator2],
    [g2_optimizer, d2_optimizer],
    [g2_scheduler, d2_scheduler],
    device,
    desc='Epoch loop G2',
)