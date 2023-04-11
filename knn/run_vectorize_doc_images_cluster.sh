MNT_DIR_PATH=${MNT_DIR_PATH:-/mnt/home/divashkov}
START_PART_NUMBER="$1"
END_PART_NUMBER="$2"

echo "cd $MNT_DIR_PATH/sd-image-to-prompts/"
cd $MNT_DIR_PATH/sd-image-to-prompts/;

echo "Starting python vectorize_doc_images.py"
$PYTHON_EXECUTABLE vectorize_doc_images.py \
--model_config_path logs/vqgan_imagenet_f16_16384/configs/model.yaml \
--model_ckpt_path logs/vqgan_imagenet_f16_16384/checkpoints/last.ckpt \
--index_path index.csv \
--data_path $MNT_DIR_PATH/data/diffusiondb_img \
--memmap_path $MNT_DIR_PATH/data/diffusiondb_memmaps/doc_images.dat \
--start_part_number $START_PART_NUMBER \
--end_part_number $END_PART_NUMBER \
--batch_size 32 \
--n_workers 8