source .env

app_data_path=$(xcrun simctl get_app_container booted com.jonbrennecke.BackflipTransmogrifier data)

app_images_path=$app_data_path/Documents

app_images_count=$(ls $app_images_path | wc -l | tr -d '[:space:]')

training_images_path=${DATA_DIR_PATH}/training_images

echo "Clearing training images from ${training_images_path}"

rm -rf $training_images_path/*

echo "Importing ${app_images_count} images to ${DATA_DIR_PATH}"

cp -R $app_images_path/* $training_images_path

