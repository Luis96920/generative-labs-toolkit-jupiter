import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm
import time
import os

from models.discriminators import MultiscaleDiscriminator
from models.generators import GlobalGenerator, LocalEnhancer
from models.loss import Loss
from models.models_utils import Encoder
from utils.dataloader import SwordSorceryDataset
from utils.utils import print_device_name, save_tensor_images

# Parse torch version for autocast
# ######################################################
version = torch.__version__
version = tuple(int(n) for n in version.split('.')[:-1])
has_autocast = version >= (1, 6)
# ######################################################

#def train(dataloader, models, optimizers, schedulers, device, desc, saved_images_path, saved_model_path, display_step=100, checkpoint=None, verbose=False, epochs=100):
def train(dataloader, models, optimizers, schedulers, args, desc=''):

    encoder, generator, discriminator = models
    g_optimizer, d_optimizer = optimizers
    g_scheduler, d_scheduler = schedulers

    loss_fn = Loss(device=args.device)

    # output paths
    saved_images_path = os.path.join(args.output_path_dir, args.saved_images_path)
    saved_model_path = os.path.join(args.output_path_dir, args.saved_model_path)

    # running variables
    cur_step = 0
    mean_g_loss = 0.0
    mean_d_loss = 0.0
    epoch_run=0

    # recover from checkpoint
    path_bkp_model = os.path.join(saved_model_path, 'bkp_model_ft.pth')
    if(args.continue_training and os.path.exists(path_bkp_model)):
        cp = torch.load(path_bkp_model)
        epoch_run = cp['epoch']
        encoder.load_state_dict(cp['encoder_state_dict'])          # Load state of the last epoch
        generator.load_state_dict(cp['generator_state_dict'])
        discriminator.load_state_dict(cp['discriminator_state_dict'])
        #best_model_wts = cp['best_model_wts']
        g_optimizer.load_state_dict(cp['g_optimizer_state_dict'])        
        d_optimizer.load_state_dict(cp['d_optimizer_state_dict'])
        g_scheduler.load_state_dict(cp['g_scheduler_state_dict'])     
        d_scheduler.load_state_dict(cp['d_scheduler_state_dict'])  
        print('Resuming script in epoch {}.'.format(epoch_run))     

    for epoch in tqdm(range(args.epochs-epoch_run), desc=desc, leave=True):
        # Training epoch
        # time
        since_load = time.time()
        for (x_real, labels, insts, bounds) in tqdm(dataloader, desc=f'  inner loop for epoch {epoch+epoch_run}'):
            x_real = x_real.to(args.device)
            labels = labels.to(args.device)
            insts = insts.to(args.device)
            bounds = bounds.to(args.device)

            # time
            time_elapsed_load = time.time() - since_load
            since_training = time.time()

            # Enable autocast to FP16 tensors (new feature since torch==1.6.0)
            # If you're running older versions of torch, comment this out
            # and use NVIDIA apex for mixed/half precision training
            if has_autocast:
                with torch.cuda.amp.autocast(enabled=(args.device=='cuda')):
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

            mean_g_loss += g_loss.item() / args.display_step
            mean_d_loss += d_loss.item() / args.display_step

            if cur_step % args.display_step == 0 and cur_step > 0:
                print('Step {}: Generator loss: {:.5f}, Discriminator loss: {:.5f}'
                      .format(cur_step, mean_g_loss, mean_d_loss))
                save_tensor_images(x_fake.to(x_real.dtype), x_real, epoch+epoch_run, cur_step, saved_images_path)
                mean_g_loss = 0.0
                mean_d_loss = 0.0
            cur_step += 1

            # time 
            time_elapsed_training = time.time() - since_training
            since_load = time.time()
            if args.verbose:
                print('Loading images complete in {:.0f}m {:.0f}s {:.0f}ms'.format(
                time_elapsed_load // 60, time_elapsed_load % 60, 60*time_elapsed_load % 60))
                print('Training complete in {:.0f}m {:.0f}s {:.0f}ms'.format(
                time_elapsed_training // 60, time_elapsed_training % 60, 60*time_elapsed_training % 60))

        g_scheduler.step()
        d_scheduler.step()

        # Save checkpoint
        if saved_model_path is not None:
            torch.save({
                'epoch': epoch + epoch_run + 1,
                # Networks states
                'encoder_state_dict': encoder.state_dict(),
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                # Best models states
                # ver! best model and best kpi
                # Optimizer states
                'g_optimizer_state_dict': g_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
                # Scheduler states
                'g_scheduler_state_dict': g_scheduler.state_dict(),
                'd_scheduler_state_dict': d_scheduler.state_dict(),
            }, path_bkp_model)


def weights_init(m):
    ''' Function for initializing all model weights '''
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0., 0.02)


def train_networks(args):

    # init params
    n_classes = args.n_classes                  # total number of object classes
    rgb_channels = n_features = args.n_features       
    
    if args.device not in {'cuda', 'cpu'}:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_dir = [{
        'path_root': args.input_path_dir, #
        'path_inputs': [(args.input_img_dir, 'orig_img'), (args.input_inst_dir, 'inst_map'), (args.input_label_dir, 'label_map')]
        }]

    # functions
    def lr_lambda(epoch):
        ''' Function for scheduling learning '''
        return 1. if epoch < args.decay_after else 1 - float(epoch - args.decay_after) / (args.num_epochs - args.decay_after)

    ### Init train
    ## Phase 1: Low Resolution (1024 x 512)
    dataloader1 = DataLoader(
        SwordSorceryDataset(train_dir, target_width=args.target_width_1, n_classes=n_classes, n_inputs=3),
        collate_fn=SwordSorceryDataset.collate_fn, batch_size=args.batch_size_1, shuffle=True, drop_last=False, pin_memory=True,
    )
    encoder = Encoder(rgb_channels, n_features).to(args.device).apply(weights_init)
    #generator1 = GlobalGenerator(n_classes + n_features + 1, rgb_channels).to(args.device).apply(weights_init)
    #discriminator1 = MultiscaleDiscriminator(n_classes + 1 + rgb_channels, n_discriminators=2).to(args.device).apply(weights_init)
    #HARCODED BECAUSE LABEL IMAGE!  n_classes=1
    generator1 = GlobalGenerator(1 + n_features + 1, rgb_channels).to(args.device).apply(weights_init)
    discriminator1 = MultiscaleDiscriminator(1 + 1 + rgb_channels, n_discriminators=2).to(args.device).apply(weights_init)

    g1_optimizer = torch.optim.Adam(list(generator1.parameters()) + list(encoder.parameters()), lr=args.lr, betas=(args.beta_1, args.beta_2))
    d1_optimizer = torch.optim.Adam(list(discriminator1.parameters()), lr=args.lr, betas=(args.beta_1, args.beta_2))
    g1_scheduler = torch.optim.lr_scheduler.LambdaLR(g1_optimizer, lr_lambda)
    d1_scheduler = torch.optim.lr_scheduler.LambdaLR(d1_optimizer, lr_lambda)


    ## Phase 2: High Resolution (2048 x 1024)
    dataloader2 = DataLoader(
        SwordSorceryDataset(train_dir, target_width=args.target_width_1, n_classes=n_classes, n_inputs=3),
        collate_fn=SwordSorceryDataset.collate_fn, batch_size=args.batch_size_2, shuffle=True, drop_last=False, pin_memory=True,
    )
    #generator2 = LocalEnhancer(n_classes + n_features + 1, rgb_channels).to(args.device).apply(weights_init)
    #discriminator2 = MultiscaleDiscriminator(n_classes + 1 + rgb_channels).to(args.device).apply(weights_init)
    #HARCODED BECAUSE LABEL IMAGE!  n_classes=1
    generator2 = LocalEnhancer(1 + n_features + 1, rgb_channels).to(args.device).apply(weights_init)
    discriminator2 = MultiscaleDiscriminator(1 + 1 + rgb_channels).to(args.device).apply(weights_init)

    g2_optimizer = torch.optim.Adam(list(generator2.parameters()) + list(encoder.parameters()), lr=args.lr, betas=(args.beta_1, args.beta_2))
    d2_optimizer = torch.optim.Adam(list(discriminator2.parameters()), lr=args.lr, betas=(args.beta_1, args.beta_2))
    g2_scheduler = torch.optim.lr_scheduler.LambdaLR(g2_optimizer, lr_lambda)
    d2_scheduler = torch.optim.lr_scheduler.LambdaLR(d2_optimizer, lr_lambda)

    ### Training
    if args.device=='cuda':
        print_device_name()

    # Phase 1: Low Resolution
    #######################################################################
    train(
        dataloader1,
        [encoder, generator1, discriminator1],
        [g1_optimizer, d1_optimizer],
        [g1_scheduler, d1_scheduler],
        args,
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
        args,
        desc='Epoch loop G2',
    )