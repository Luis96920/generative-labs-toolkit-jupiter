import argparse
import os
import warnings
from training.training import train_networks
from utils.utils import str2bool
from datetime import datetime
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def parse_args():
    desc = "Pix2PixHD"

    parser = argparse.ArgumentParser(description=desc)

    # Dataset parameters and input paths
    parser.add_argument('--n_classes', type=int, default=2, help='Number of segmented instances in the dataset. Eg. Character and background')
    parser.add_argument('--n_features', type=int, default=3, help='Number of channels. Eg. 3 for RGB')
    parser.add_argument('--input_path_dir', type=str, default="/Users/ramirocasal/Documents/Datasets/sword_sorcery_data_for_ramiro/test_dataset", help='Path root where inputs are located. By default it will contain 3 subfolders: img, inst, label')
    parser.add_argument('--input_img_dir', type=str, default="02_output", help='Folder name for input images located in input_path_dir')
    parser.add_argument('--output_img_dir', type=str, default="02_output", help='Folder name for output images located in input_path_dir')
    parser.add_argument('--input_label_dir', type=str, default="--nodir--", help='Folder name for optional labeled images located in input_path_dir')        # 01_segmented_input
    parser.add_argument('--input_inst_dir', type=str, default="01_segmented_input", help='Folder name for optional instances images located in input_path_dir') 

    # Training parameters
    parser.add_argument('--epochs1', type=int, default=20, help='The number of epochs step 1')   
    parser.add_argument('--epochs2', type=int, default=20, help='The number of epochs step 2')   
    parser.add_argument('--decay_after', type=int, default=200, help='Number of epochs with constant lr')   
    parser.add_argument('--lr', type=int, default=0.0005, help='Learning rate')   
    parser.add_argument('--beta_1', type=int, default=0.5, help='Parameter beta_1 for Adam optimizer') 
    parser.add_argument('--beta_2', type=int, default=0.999, help='Parameter beta_2 for Adam optimizer') 
    parser.add_argument('--batch_size_1', type=int, default=2, help='The size of batch for stage 1 of training.')
    parser.add_argument('--batch_size_2', type=int, default=1, help='The size of batch for stage 2 of training.')
    parser.add_argument('--target_width_1', type=int, default=1024, help='The size of image for stage 1 of training.')
    parser.add_argument('--target_width_2', type=int, default=2048, help='The size of image for stage 2 of training.')

    # Experiment parameters
    parser.add_argument('--experiment_name', type=str, default="", help='A name for the experiment')
    parser.add_argument('--verbose', type=int, default=0, help='Display training time metrics. Yes: 1, No: 2')
    parser.add_argument('--display_step', type=int, default=100, help='Number of step to display images.')
    parser.add_argument('--resume_training', type=str2bool, nargs='?', const=True, default=False, help="Continue training allows to resume training. You'll need to add experiment name args to identify the experiment to recover.")

    # Output paths
    parser.add_argument('--output_path_dir', type=str, default="", help='The base directory to hold the results')
    parser.add_argument('--saved_images_path', type=str, default="Images", help='Folder name for save images during training')
    parser.add_argument('--saved_model_path', type=str, default="Saved_Models", help='Folder name for save model')

    # Distributed configuration 
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N', help='Number of nodes')
    parser.add_argument('-g', '--gpus', default=1, type=int, help='Number of GPUs per node')
    parser.add_argument('-nr', '--nr', default=0, type=int, help='Ranking within the nodes')
    if dist.is_available():
        parser.add_argument('--backend', type=str, help='distributed backend',
                        choices=[dist.Backend.GLOO, dist.Backend.NCCL, dist.Backend.MPI],
                        default=dist.Backend.NCCL)

    # Warnings parameters
    parser.add_argument('--warnings', type=str2bool, nargs='?', const=False, default=True, help="Show warnings")


    """ 
    parser.add_argument('--history_dir', type=str, default="History/", help='The directory for input data')
    parser.add_argument('--notes', type=str, default="N/A", help='A description of the experiment')
    """
    return parser.parse_args()


def main():
    args = parse_args()

    # warnings
    if args.warnings:
        warnings.filterwarnings("ignore")

    # Resume training and experiment name
    if (args.resume_training):
        args.experiment_name = args.experiment_name
        args.low_resolution_finished = torch.load(os.path.join(args.output_path_dir, args.experiment_name, args.saved_model_path, 'training_status.info'))['low_resolution_finished']
    else:
        args.experiment_name = datetime.now().strftime("%Y_%m_%d_%H_%M") + "_" + args.experiment_name
        args.low_resolution_finished = False

    # Output path dir
    args.output_path_dir = os.path.join(args.output_path_dir,args.experiment_name) 
    if(not os.path.exists(args.output_path_dir)):
        print('creating directories in ' + args.output_path_dir)
        os.makedirs(args.output_path_dir)
        os.makedirs(os.path.join(args.output_path_dir, args.saved_images_path))
        os.makedirs(os.path.join(args.output_path_dir,"History"))
        os.makedirs(os.path.join(args.output_path_dir, args.saved_model_path))

    # Multiprocessing
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8888'
    mp.spawn(train_networks, nprocs=args.gpus, args=(args,))   
    
    #train_networks(args)
    print("done training")


if __name__ == '__main__':
    main()

