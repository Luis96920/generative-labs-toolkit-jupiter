python3 pix2pixhd/pix2pixhd.py \
   --input_path_dir "/home/jupyter/Ramiro/dataset/unet_data/full" \
   --input_img_dir "X" \
   --output_img_dir "Y" \
   --epochs1 1 \
   --epochs2 1 \
   --display_step 1000 \
   --output_path_dir "/home/jupyter/Ramiro/Results" \
   --target_width_1 512 \
   --target_width_2 1024 \
   --batch_size_1 10 \
   --gpus 2 \
   --experiment_name "testing_exp" \
    #--resume_training \