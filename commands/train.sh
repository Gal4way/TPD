# VITONHD_release_person_combine_garment_240epochs
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--logdir train_logs/VITONHD/ \
--pretrained_model checkpoints/original/model_prepared.ckpt \
--base configs/train/train_VITONHD.yaml \
--scale_lr False \
--name VITONHD_release_person_combine_garment_240epochs