MNT_DIR_PATH=${MNT_DIR_PATH:-/mnt/home/}

echo "cd $MNT_DIR_PATH/sd-image-to-prompts/"
cd $MNT_DIR_PATH/sd-image-to-prompts/;

echo "Starting python vectorize_query_images.py"
$PYTHON_EXECUTABLE vectorize_query_images.py \
--query_memmap_path=$MNT_DIR_PATH/data/diffusiondb_memmaps/query_images.dat \
--doc_memmap_path=$MNT_DIR_PATH/data/diffusiondb_memmaps/doc_images.dat \
--doc_prompts_path=/data/home/repos/sd-image-to-prompts/index.csv \
--query_meta_path=/mnt/home/sd-image-to-prompts/test_submit_dataset/prompts.csv \
--doc_memmap_length=14000000 \
--query_memmap_length=7 \
--doc_memmap_dim=1024 \
--query_memmap_dim=1024 \
--use_gpu=true
