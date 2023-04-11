MNT_DIR_PATH=${MNT_DIR_PATH:-/mnt/home/}

echo "cd $MNT_DIR_PATH/sd-image-to-prompts/"
cd $MNT_DIR_PATH/sd-image-to-prompts/;

echo "Starting python vectorize_query_images.py"
$PYTHON_EXECUTABLE vectorize_query_images.py \
--model_config_path logs/vqgan_imagenet_f16_16384/configs/model.yaml \
--model_ckpt_path logs/vqgan_imagenet_f16_16384/checkpoints/last.ckpt \
--index_path test_submit_dataset/prompts.csv \
--data_path test_submit_dataset/images/ \
--memmap_path $MNT_DIR_PATH/data/diffusiondb_memmaps/query_images.dat \
--batch_size 32 \
--n_workers 8