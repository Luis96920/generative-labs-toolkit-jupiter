import argparse
import os
import shutil
from training.training import train_networks
from datetime import datetime


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
    parser.add_argument('--num_epochs', type=int, default=200, help='The number of epochs')   
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

    # Output paths
    parser.add_argument('--output_path_dir', type=str, default="", help='The base directory to hold the results')
    parser.add_argument('--saved_images_path', type=str, default="Images", help='Folder name for save images during training')




    """ 
    parser.add_argument('--input_img_data_dir', type=str, default="Image_X/", help='The directory for CSV input data')
    parser.add_argument('--output_data_dir', type=str, default="CSV_Y/", help='The directory for CSV input data')
    parser.add_argument('--base_results_dir', type=str, default="/", help='The base directory to hold the results')
    parser.add_argument('--output_test_csv_dir', type=str, default="CSV/", help='The directory for result csvs')
    parser.add_argument('--output_test_graph_dir', type=str, default="Graph/", help='The directory for result csvs')
    parser.add_argument('--saved_model_dir', type=str, default="Saved_Models/", help='The directory for input data')
    parser.add_argument('--history_dir', type=str, default="History/", help='The directory for input data')
    parser.add_argument('--csv_dims', type=int, default=156, help='The number of csv channels')
    parser.add_argument('--input_channels', type=int, default=2, help='The number of input bone dims')
    parser.add_argument('--output_channels', type=int, default=2, help='The number of input bone dims')
    parser.add_argument('--latent_dim', type=int, default=64, help='the size of the latent dim')
    parser.add_argument('--print_freq', type=int, default=5, help='How often is the status printed')
    parser.add_argument('--save_freq', type=int, default=10, help='How often is the model saved')
    parser.add_argument('--save_best_only', action='store_true')
    parser.add_argument('--print_csv', action='store_true')
    parser.add_argument('--continue_training', action='store_true')
    parser.add_argument('--saved_weights_path', type=str, default="N/A", help='the path of the saved weights')
    parser.add_argument('--notes', type=str, default="N/A", help='A description of the experiment')
    parser.add_argument('--img_2_bone', action='store_true')
    """
    return parser.parse_args()


def main():
    args = parse_args()

    args.experiment_name = datetime.now().strftime("%Y_%m_%d_%H_%M") + "_" + args.experiment_name
    args.base_results_dir = os.path.join(args.base_results_dir,args.experiment_name)
    
    if(not os.path.exists(args.base_results_dir)):
        print('creating directories in ' + args.base_results_dir)
        os.makedirs(args.base_results_dir)
        os.makedirs(os.path.join(args.base_results_dir, args.saved_images_path))
        os.makedirs(os.path.join(args.base_results_dir,"History"))
        os.makedirs(os.path.join(args.base_results_dir,"Saved_Models"))

    """
    if(args.img_2_bone):
        normalize_image_data(args)
        train_img_2_bone(args)
    else:
        normalize_data(args)
        train(args)
        
    """

    train_networks(args)
    print("done training")


if __name__ == '__main__':
    main()

