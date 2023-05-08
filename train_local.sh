# the only thing you need is to add CONFIG_NAME env var
PYTHON_EXECUTABLE=python
MNT_DIR_PATH=/mnt/home/
REPO_NAME=sd_t
SCRIPT_NAME=train_blip.py
CONFIG_DIR=/mnt/home/sd_t/
CONFIG_NAME=train

cd $MNT_DIR_PATH/$REPO_NAME/;
CONFIG_DIR=$CONFIG_DIR \
CONFIG_NAME=$CONFIG_NAME \
$PYTHON_EXECUTABLE $SCRIPT_NAME \
++mnt_dir_path=$MNT_DIR_PATH \
"$@"