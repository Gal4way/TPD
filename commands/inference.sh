# VITONHD_release_input_person_combine_garment_240epochs_paired
CUDA_VISIBLE_DEVICES=0 python scripts/inference.py \
--config configs/inference/inference_VITONHD_paired.yaml \
--ckpt checkpoints/release/TPD_240epochs.ckpt \
--outdir inference_logs/VITONHD/VITONHD_release_input_person_combine_garment_240epochs_paired/ \
--seed 321 \
--batch_size 15 \
--predicted_mask_dilation 0 \
--C 5 \
--H 512 \
--W 768 \


# VITONHD_release_input_person_combine_garment_240epochs_unpaired
CUDA_VISIBLE_DEVICES=1 python scripts/inference.py \
--config configs/inference/inference_VITONHD_unpaired.yaml \
--ckpt checkpoints/release/TPD_240epochs.ckpt \
--outdir inference_logs/VITONHD/VITONHD_release_input_person_combine_garment_240epochs_unpaired/ \
--seed 321 \
--batch_size 15 \
--C 5 \
--H 512 \
--W 768 \