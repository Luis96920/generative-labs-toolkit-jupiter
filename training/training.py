import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist

from tqdm import tqdm
import time
import os

from models.discriminators import MultiscaleDiscriminator
from models.generators import GlobalGenerator, LocalEnhancer
from models.loss import gd_loss, VGG_Loss
from models.models_utils import Encoder
from utils.dataloader import create_loaders
from utils.utils import save_tensor_images, should_distribute, is_distributed

import ray
import ray.train as train
from ray.util.sgd.torch import TrainingOperator
from ray.util.sgd import TorchTrainer

NODES      = int(os.environ.get('WORLD_SIZE', 1))

# Parse torch version for autocast
# ######################################################
version = torch.__version__
version = tuple(int(n) for n in version.split('.')[:-1])
has_autocast = version >= (1, 6)
# ######################################################

def train_networks(gpu, args):
    
    # init params
    n_classes = args.n_classes                  # total number of object classes # maybe we're not using n_classes, just args.n_classes
    rgb_channels = n_features = args.n_features       
    args.gpu = gpu

    # Device conf, GPU and distributed computing
    torch.cuda.set_device(gpu)

    # Multiprocesing
    rank = args.nr * args.gpus + gpu
    if should_distribute(args.world_size):
        print('Using distributed PyTorch with {} backend'.format(args.backend))
        dist.init_process_group(backend=args.backend, init_method='env://', world_size=args.world_size, rank=rank)
    
    # Training directories
    train_dir = {
        'path_root': args.input_path_dir, 
        'path_inputs': {
            'input_img': args.input_img_dir,
            'inst_map': args.input_inst_dir,
            'label_map': args.input_label_dir, 
            'output_img': args.output_img_dir
            }
        }

    # functions
    def lr_lambda(epoch):
        ''' Function for scheduling learning '''
        return 1. if epoch < args.decay_after else 1 - float(epoch - args.decay_after) / (args.num_epochs - args.decay_after)

    ### Init train
    ## Phase 1: Low Resolution (1024 x 512)
    dataloader1 = create_loaders(train_dir, target_width=args.target_width_1, batch_size=args.batch_size_1, n_classes=n_classes, world_size=args.world_size, rank=rank)
    encoder = Encoder(rgb_channels, n_features).cuda(gpu).apply(weights_init)
    generator1 = GlobalGenerator(dataloader1.dataset.get_input_size_g(), rgb_channels).cuda(gpu).apply(weights_init)
    discriminator1 = MultiscaleDiscriminator(dataloader1.dataset.get_input_size_d(), n_discriminators=2).cuda(gpu).apply(weights_init)

    g1_optimizer = torch.optim.Adam(list(generator1.parameters()) + list(encoder.parameters()), lr=args.lr, betas=(args.beta_1, args.beta_2))
    d1_optimizer = torch.optim.Adam(list(discriminator1.parameters()), lr=args.lr, betas=(args.beta_1, args.beta_2))
    g1_scheduler = torch.optim.lr_scheduler.LambdaLR(g1_optimizer, lr_lambda)
    d1_scheduler = torch.optim.lr_scheduler.LambdaLR(d1_optimizer, lr_lambda)

    ## Phase 2: High Resolution (2048 x 1024)
    dataloader2 = create_loaders(train_dir, target_width=args.target_width_2, batch_size=args.batch_size_2, n_classes=n_classes, world_size=args.world_size, rank=rank)
    generator2 = LocalEnhancer(dataloader2.dataset.get_input_size_g(), rgb_channels).cuda(gpu).apply(weights_init)
    discriminator2 = MultiscaleDiscriminator(dataloader2.dataset.get_input_size_d()).cuda(gpu).apply(weights_init)

    g2_optimizer = torch.optim.Adam(list(generator2.parameters()) + list(encoder.parameters()), lr=args.lr, betas=(args.beta_1, args.beta_2))
    d2_optimizer = torch.optim.Adam(list(discriminator2.parameters()), lr=args.lr, betas=(args.beta_1, args.beta_2))
    g2_scheduler = torch.optim.lr_scheduler.LambdaLR(g2_optimizer, lr_lambda)
    d2_scheduler = torch.optim.lr_scheduler.LambdaLR(d2_optimizer, lr_lambda)

    if is_distributed():
        encoder = nn.parallel.DistributedDataParallel(encoder, device_ids=[gpu])
        generator1 = nn.parallel.DistributedDataParallel(generator1, device_ids=[gpu])
        discriminator1 = nn.parallel.DistributedDataParallel(discriminator1, device_ids=[gpu])
        generator2 = nn.parallel.DistributedDataParallel(generator2, device_ids=[gpu])
        discriminator2 = nn.parallel.DistributedDataParallel(discriminator2, device_ids=[gpu])


    ### Training
    # output paths
    args.saved_images_path = os.path.join(args.output_path_dir, args.saved_images_path)
    args.saved_model_path = os.path.join(args.output_path_dir, args.saved_model_path)

    # Phase 1: Low Resolution
    #######################################################################
    torch.save({'low_resolution_finished': False}, os.path.join(args.saved_model_path, 'training_status.info'))
    
    if not args.low_resolution_finished:
        train(
            dataloader1,
            [encoder, generator1, discriminator1],
            [g1_optimizer, d1_optimizer],
            [g1_scheduler, d1_scheduler],
            args,
            stage='stage1',
            desc='Epoch loop G1',
        )

    torch.save({'low_resolution_finished': True}, os.path.join(args.saved_model_path, 'training_status.info'))

    # Phase 2: High Resolution
    #######################################################################
    # Update global generator in local enhancer with trained
    generator2.g1 = generator1.g1

    # delete models from GPU for releasing space
    del discriminator1, g1_optimizer, d1_optimizer, g1_scheduler, d1_scheduler
    torch.cuda.empty_cache()

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
        stage='stage2',
        desc='Epoch loop G2',
    )


def train(dataloader, models, optimizers, schedulers, args, stage='', desc=''):

    encoder, generator, discriminator = models
    g_optimizer, d_optimizer = optimizers
    g_scheduler, d_scheduler = schedulers

    vgg_loss = VGG_Loss(gpu=args.gpu)

    # running variables
    cur_step = 0
    mean_g_loss = 0.0
    mean_d_loss = 0.0
    epoch_run=0

    # recover from checkpoint
    path_bkp_model = os.path.join(args.saved_model_path, 'bkp_model_' + stage + '.pth')
    print('Resume training: ' + str(args.resume_training))
    if(args.resume_training and os.path.exists(path_bkp_model)):
        cp = torch.load(path_bkp_model)
        epoch_run = cp['epoch']
        #encoder.load_state_dict(cp['encoder_state_dict'])          # Load state of the last epoch
        generator.load_state_dict(cp['generator_state_dict'])
        discriminator.load_state_dict(cp['discriminator_state_dict'])
        #best_model_wts = cp['best_model_wts']
        g_optimizer.load_state_dict(cp['g_optimizer_state_dict'])        
        d_optimizer.load_state_dict(cp['d_optimizer_state_dict'])
        g_scheduler.load_state_dict(cp['g_scheduler_state_dict'])     
        d_scheduler.load_state_dict(cp['d_scheduler_state_dict'])  
        print('Resuming script in epoch {}, {}.'.format(epoch_run,stage))     

    for epoch in tqdm(range(args.epochs-epoch_run), desc=desc, leave=True):
        # Training epoch
        # time
        since_load = time.time()
        for (img_i, labels, insts, bounds, img_o) in tqdm(dataloader, desc=f'  inner loop for epoch {epoch+epoch_run}'):
            img_i = img_i.cuda(args.gpu)
            labels = labels.cuda(args.gpu)#to(args.device)
            insts = insts.cuda(args.gpu)#to(args.device)
            bounds = bounds.cuda(args.gpu)#to(args.device)
            img_o = img_o.cuda(args.gpu)#to(args.device)

            # time
            time_elapsed_load = time.time() - since_load
            since_training = time.time()

            # Enable autocast to FP16 tensors (new feature since torch==1.6.0)
            # If you're running older versions of torch, comment this out
            # and use NVIDIA apex for mixed/half precision training
            if has_autocast:
                with torch.cuda.amp.autocast(enabled=True):
                    img_o_fake, fake_preds_for_g, fake_preds_for_d, real_preds_for_d = forward_pass(
                        img_i, labels, insts, bounds, img_o, encoder, generator, discriminator)

                    g_loss, d_loss = gd_loss(fake_preds_for_g, real_preds_for_d, fake_preds_for_d, img_o_fake, img_o, discriminator.module.n_discriminators, vgg_loss)
                    img_o_fake = img_o_fake.detach()
            else:
                img_o_fake, fake_preds_for_g, fake_preds_for_d, real_preds_for_d = forward_pass(
                    img_i, labels, insts, bounds, img_o, encoder, generator, discriminator)

                g_loss, d_loss = gd_loss(fake_preds_for_g, real_preds_for_d, fake_preds_for_d, img_o_fake, img_o, discriminator.module.n_discriminators, vgg_loss)
                img_o_fake = img_o_fake.detach()

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
                save_tensor_images(img_o_fake.to(img_o.dtype), img_o, epoch+epoch_run, stage, cur_step, args.saved_images_path)
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
        if args.saved_model_path is not None:
            torch.save({
                'epoch': epoch + epoch_run + 1,
                # Networks states
                #'encoder_state_dict': encoder.state_dict(),
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



def forward_pass(img_i, label_map, instance_map, boundary_map, img_o_real, encoder, generator, discriminator):
    '''
    Function that computes the forward pass and total loss for generator and discriminator.
    '''
    #feature_map = encoder(x_real, instance_map)
    #x_fake = generator(torch.cat((label_map, boundary_map, feature_map), dim=1))
    img_o_fake = generator(torch.cat((img_i,label_map, boundary_map), dim=1))

    # Get necessary outputs for loss/backprop for both generator and discriminator
    #fake_preds_for_g = discriminator(torch.cat((label_map, boundary_map, x_fake), dim=1))
    #fake_preds_for_d = discriminator(torch.cat((label_map, boundary_map, x_fake.detach()), dim=1))
    #real_preds_for_d = discriminator(torch.cat((label_map, boundary_map, x_real.detach()), dim=1))
    fake_preds_for_g = discriminator(torch.cat((boundary_map,label_map, img_o_fake), dim=1))
    fake_preds_for_d = discriminator(torch.cat((boundary_map,label_map, img_o_fake.detach()), dim=1))
    real_preds_for_d = discriminator(torch.cat((boundary_map,label_map, img_o_real.detach()), dim=1))

    return img_o_fake, fake_preds_for_g, fake_preds_for_d, real_preds_for_d





### Train with RAY
def train_ray(dataloader, models, optimizers, schedulers, args, stage='', desc=''):
    config = {}
    config['dataloader']=dataloader
    config['models']=models
    config['optimizers']=optimizers
    config['schedulers']=schedulers
    config['args']=args
    config['lr']=args.lr
    config['batch_size']=args.batch_size_1


    ray.init()

    trainer = TorchTrainer(
        training_operator_cls=Pix2PixHDOperator,
        num_workers=1,
        config=config,
        use_gpu=True
    )

    for i in range(args.epochs):
        stats = trainer.train()
        print(stats)


class Pix2PixHDOperator(TrainingOperator):
    def setup(self, config):
        """Setup for this operator.

        This is where you define the training state and register it with Ray SGD.

        Args:
            config (dict): Custom configuration value to be passed to
                all creator and operator constructors. Same as ``self.config``.
        """
        dataloader = config["dataloader"] 
        models = config["models"]
        optimizers=config["optimizers"]
        schedulers=config["schedulers"] 
        args = config["args"]

        self.args = args   
        self.models, self.optimizers, self.criterion = \
            self.register(models=models, optimizers=optimizers,
                          criterion=gd_loss,
                          #schedulers=schedulers
                          )
        self.register_data(train_loader=dataloader)

    def train_batch(self, batch, batch_info):
        """Trains on one batch of data from the data creator.
        Args:
            batch: One item of the validation iterator.
            batch_info (dict): Information dict passed in from ``train_epoch``.

        Returns:
            A dict of metrics. Defaults to "loss" and "num_samples",
                corresponding to the total number of datapoints in the batch.
        """
        #Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        encoder, generator, discriminator = self.models
        g_optimizer, d_optimizer = self.optimizers

        #loss_fn = Loss(device=args.device)
        vgg_loss = VGG_Loss(device=self.args.device)

        #for (img_i, labels, insts, bounds, img_o) in tqdm(dataloader, desc=f'  inner loop for epoch {epoch+epoch_run}'):
        img_i = batch.img_i.to(self.args.device)
        labels = batch.labels.to(self.args.device)
        insts = batch.insts.to(self.args.device)
        bounds = batch.bounds.to(self.args.device)
        img_o = batch.img_o.to(self.args.device)

        img_o_fake, fake_preds_for_g, fake_preds_for_d, real_preds_for_d = forward_pass(
            img_i, labels, insts, bounds, img_o, encoder, generator, discriminator)

        g_loss, d_loss = gd_loss(fake_preds_for_g, real_preds_for_d, fake_preds_for_d, img_o_fake, img_o, discriminator.n_discriminators, vgg_loss)
        img_o_fake = img_o_fake.detach()


        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        return {
                "loss_g": g_loss.item(),
                "loss_d": d_loss.item(),
                "num_samples": batch.img_i.shape[0]
            }




