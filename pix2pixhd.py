import argparse
import os
import shutil
from training.training import train_networks
from utils.utils import str2bool
from datetime import datetime
import torch

def parse_args():
    desc = "An autoencoder for pose similarity detection"

    parser = argparse.ArgumentParser(description=desc)

    # Dataset parameters and input paths
    parser.add_argument('--n_classes', type=int, default=2, help='Number of segmented instances in the dataset. Eg. Character and background')
    parser.add_argument('--n_features', type=int, default=3, help='Number of channels. Eg. 3 for RGB')
    parser.add_argument('--input_path_dir', type=str, default="/content/drive/MyDrive/GenerativeLabs/dataset/sword_sorcery_data_for_ramiro/paired_data", help='Path root where inputs are located. By default it will contain 3 subfolders: img, inst, label')
    parser.add_argument('--input_img_dir', type=str, default="02_output", help='Folder name for original images located in input_path_dir')
    parser.add_argument('--input_label_dir', type=str, default="01_segmented_input", help='Folder name for labeled images located in input_path_dir')        
    parser.add_argument('--input_inst_dir', type=str, default="01_segmented_input", help='Folder name for instances images located in input_path_dir') 

    # Training parameters
    parser.add_argument('--epochs', type=int, default=200, help='The number of epochs')   
    parser.add_argument('--decay_after', type=int, default=200, help='Number of epochs with constant lr')   
    parser.add_argument('--lr', type=int, default=0.0005, help='Learning rate')   
    parser.add_argument('--beta_1', type=int, default=0.5, help='Parameter beta_1 for Adam optimizer') 
    parser.add_argument('--beta_2', type=int, default=0.999, help='Parameter beta_2 for Adam optimizer') 
    parser.add_argument('--batch_size_1', type=int, default=2, help='The size of batch for step 1 of training.')
    parser.add_argument('--batch_size_2', type=int, default=1, help='The size of batch for step 2 of training.')
    parser.add_argument('--target_width_1', type=int, default=1024, help='The size of image for step 1 of training.')
    parser.add_argument('--target_width_2', type=int, default=1024, help='The size of image for step 1 of training.')

    # Experiment parameters
    parser.add_argument('--experiment_name', type=str, default="", help='A name for the experiment')
    parser.add_argument('--verbose', type=int, default=0, help='Display training time metrics. Yes: 1, No: 2')
    parser.add_argument('--display_step', type=int, default=100, help='Number of step to display images.')
    parser.add_argument('--device', type=str, default="auto", help='Device for training network. Options cpu, cuda or auto')
    parser.add_argument('--continue_training', type=str2bool, nargs='?', const=True, default=False, help="Continue training allows to resume training. You'll need to add experiment name args to identify the experiment to recover.")

    # Output paths
    parser.add_argument('--output_path_dir', type=str, default="", help='The base directory to hold the results')
    parser.add_argument('--saved_images_path', type=str, default="Images", help='Folder name for save images during training')
    parser.add_argument('--saved_model_path', type=str, default="Saved_Models", help='Folder name for save model')


    """ 
    parser.add_argument('--history_dir', type=str, default="History/", help='The directory for input data')
    parser.add_argument('--input_channels', type=int, default=2, help='The number of input bone dims')
    parser.add_argument('--output_channels', type=int, default=2, help='The number of input bone dims')
    parser.add_argument('--latent_dim', type=int, default=64, help='the size of the latent dim')
    parser.add_argument('--notes', type=str, default="N/A", help='A description of the experiment')
    """
    return parser.parse_args()


def main():
    args = parse_args()

    if (args.continue_training):
        args.experiment_name = args.experiment_name
        args.low_resolution_finished = torch.load(os.path.join(args.output_path_dir, args.saved_images_path, 'training_status.info'))['low_resolution_finished']
    else:
        args.experiment_name = datetime.now().strftime("%Y_%m_%d_%H_%M") + "_" + args.experiment_name
        args.low_resolution_finished = False

    args.output_path_dir = os.path.join(args.output_path_dir,args.experiment_name) 
   
    if(not os.path.exists(args.output_path_dir)):
        print('creating directories in ' + args.output_path_dir)
        os.makedirs(args.output_path_dir)
        os.makedirs(os.path.join(args.output_path_dir, args.saved_images_path))
        os.makedirs(os.path.join(args.output_path_dir,"History"))
        os.makedirs(os.path.join(args.output_path_dir, args.saved_model_path))


    train_networks(args)
    print("done training")


if __name__ == '__main__':
    main()

