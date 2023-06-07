#!/usr/bin/env bash
# SET HERE the envirnonment variables!
DEVICE=0
OUTPUT_DIR='your/path/to/where/the/images/are/gonna/be/saved'
SEN12MS_ROOT='your/path/to/the/SEN12MS/dataset'

# -- BARREN samples

# Tensorboard logging
python SEN12MS_inpainting_tensorboard.py \
--gpu $DEVICE \
--output_dir $OUTPUT_DIR \
--SEN12MS_root $SEN12MS_ROOT \
--inp_classes Barren \
--perc_area_cov 90 \
--pol_bands VVVH \
--samples_per_class 400 \
--max_iter 6001 \
--loss_1 ssim \
--net Unet \
--input_depth 2 \
--pad reflection \
--upsample deconv \
--activation Tanh \
--noise_dist uniform

# Wandb logging
#python SEN12MS_inpainting_wandb.py \
#--gpu $DEVICE \
#--output_dir $OUTPUT_DIR \
#--SEN12MS_root $SEN12MS_ROOT \
#--inp_classes Barren \
#--perc_area_cov 90 \
#--pol_bands VVVH \
#--samples_per_class 400 \
#--max_iter 6001 \
#--loss_1 ssim \
#--net Unet \
#--input_depth 2 \
#--pad reflection \
#--upsample deconv \
#--activation Tanh \
#--noise_dist uniform

# -- CROPLANDS samples

# Tensorboard logging
python SEN12MS_inpainting_tensorboard.py \
--gpu $DEVICE \
--output_dir $OUTPUT_DIR \
--SEN12MS_root $SEN12MS_ROOT \
--inp_classes Croplands \
--perc_area_cov 90 \
--pol_bands VVVH \
--samples_per_class 400 \
--max_iter 6001 \
--loss_1 ssim \
--net Unet \
--input_depth 2 \
--pad reflection \
--upsample deconv \
--activation LeakyReLU \
--noise_dist uniform

# Wandb logging
#python SEN12MS_inpainting_wandb.py \
#--gpu $DEVICE \
#--output_dir $OUTPUT_DIR \
#--SEN12MS_root $SEN12MS_ROOT \
#--inp_classes Croplands \
#--perc_area_cov 90 \
#--pol_bands VVVH \
#--samples_per_class 400 \
#--max_iter 6001 \
#--loss_1 ssim \
#--net Unet \
#--input_depth 2 \
#--pad reflection \
#--upsample deconv \
#--activation LeakyReLU \
#--noise_dist uniform

# -- FOREST samples

# Tensorboard logging
python SEN12MS_inpainting_tensorboard.py \
--gpu $DEVICE \
--output_dir $OUTPUT_DIR \
--SEN12MS_root $SEN12MS_ROOT \
--inp_classes Forest \
--perc_area_cov 90 \
--pol_bands VVVH \
--samples_per_class 400 \
--max_iter 6001 \
--loss_1 ssim \
--net Unet \
--input_depth 2 \
--pad reflection \
--upsample deconv \
--activation Tanh \
--noise_dist uniform

# Wandb logging
#python SEN12MS_inpainting_wandb.py \
#--gpu $DEVICE \
#--output_dir $OUTPUT_DIR \
#--SEN12MS_root $SEN12MS_ROOT \
#--inp_classes Croplands \
#--perc_area_cov 90 \
#--pol_bands VVVH \
#--samples_per_class 400 \
#--max_iter 6001 \
#--loss_1 ssim \
#--net Unet \
#--input_depth 2 \
#--pad reflection \
#--upsample deconv \
#--activation Tanh \
#--noise_dist uniform

# -- GRASSLAND samples

# Tensorboard logging
python SEN12MS_inpainting_tensorboard.py \
--gpu $DEVICE \
--output_dir $OUTPUT_DIR \
--SEN12MS_root $SEN12MS_ROOT \
--inp_classes Grassland \
--perc_area_cov 90 \
--pol_bands VVVH \
--samples_per_class 400 \
--max_iter 6001 \
--loss_1 ssim \
--net Unet \
--input_depth 2 \
--pad reflection \
--upsample deconv \
--activation LeakyReLU \
--noise_dist uniform

# Wandb logging
#python SEN12MS_inpainting_wandb.py \
#--gpu $DEVICE \
#--output_dir $OUTPUT_DIR \
#--SEN12MS_root $SEN12MS_ROOT \
#--inp_classes Grassland \
#--perc_area_cov 90 \
#--pol_bands VVVH \
#--samples_per_class 400 \
#--max_iter 6001 \
#--loss_1 ssim \
#--net Unet \
#--input_depth 2 \
#--pad reflection \
#--upsample deconv \
#--activation LeakyReLU \
#--noise_dist uniform

# -- SHRUBLAND samples

# Tensorboard logging
python SEN12MS_inpainting_tensorboard.py \
--gpu $DEVICE \
--output_dir $OUTPUT_DIR \
--SEN12MS_root $SEN12MS_ROOT \
--inp_classes Shrubland \
--perc_area_cov 90 \
--pol_bands VVVH \
--samples_per_class 400 \
--max_iter 6001 \
--loss_1 ssim \
--net Unet \
--input_depth 2 \
--pad reflection \
--upsample deconv \
--activation ReLU \
--noise_dist uniform

# Wandb logging
#python SEN12MS_inpainting_wandb.py \
#--gpu $DEVICE \
#--output_dir $OUTPUT_DIR \
#--SEN12MS_root $SEN12MS_ROOT \
#--inp_classes Shrubland \
#--perc_area_cov 90 \
#--pol_bands VVVH \
#--samples_per_class 400 \
#--max_iter 6001 \
#--loss_1 ssim \
#--net Unet \
#--input_depth 2 \
#--pad reflection \
#--upsample deconv \
#--activation ReLU \
#--noise_dist uniform

# -- URBAN samples

# Tensorboard logging
python SEN12MS_inpainting_tensorboard.py \
--gpu $DEVICE \
--output_dir $OUTPUT_DIR \
--SEN12MS_root $SEN12MS_ROOT \
--inp_classes Urban \
--perc_area_cov 90 \
--pol_bands VVVH \
--samples_per_class 400 \
--max_iter 6001 \
--loss_1 ssim \
--net Unet \
--input_depth 2 \
--pad reflection \
--upsample deconv \
--activation ReLU \
--noise_dist uniform

# Wandb logging
#python SEN12MS_inpainting_wandb.py \
#--gpu $DEVICE \
#--output_dir $OUTPUT_DIR \
#--SEN12MS_root $SEN12MS_ROOT \
#--inp_classes Urban \
#--perc_area_cov 90 \
#--pol_bands VVVH \
#--samples_per_class 400 \
#--max_iter 6001 \
#--loss_1 ssim \
#--net Unet \
#--input_depth 2 \
#--pad reflection \
#--upsample deconv \
#--activation ReLU \
#--noise_dist uniform

# -- WATER samples

# Tensorboard logging
python SEN12MS_inpainting_tensorboard.py \
--gpu $DEVICE \
--output_dir $OUTPUT_DIR \
--SEN12MS_root $SEN12MS_ROOT \
--inp_classes Water \
--perc_area_cov 90 \
--pol_bands VVVH \
--samples_per_class 400 \
--max_iter 6001 \
--loss_1 ssim \
--net Unet \
--input_depth 2 \
--pad reflection \
--upsample deconv \
--activation Tanh \
--noise_dist uniform

# Wandb logging
#python SEN12MS_inpainting_wandb.py \
#--gpu $DEVICE \
#--output_dir $OUTPUT_DIR \
#--SEN12MS_root $SEN12MS_ROOT \
#--inp_classes Water \
#--perc_area_cov 90 \
#--pol_bands VVVH \
#--samples_per_class 400 \
#--max_iter 6001 \
#--loss_1 ssim \
#--net Unet \
#--input_depth 2 \
#--pad reflection \
#--upsample deconv \
#--activation Tanh \
#--noise_dist uniform