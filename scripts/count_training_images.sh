source .env

training_images_path=${DATA_DIR_PATH}/training_images

training_images_count=$(ls $training_images_path | wc -l | tr -d '[:space:]')

echo "Number of training images: ${training_images_count}"
