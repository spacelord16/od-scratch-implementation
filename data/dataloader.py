import os
import tensorflow as tf 
from config.config import Config
# from od-scratch.data.dataloader import load_dataset

def load_and_preprocess_images(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=Config.CHANNELS)
    image = tf.image.resize(image, [Config.IMG_HEIGHT, Config.IMG_WIDTH])
    image /= 255.0
    return image

def load_dataset(data_dir):
    image_path = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir)]
    dataset = tf.data.Dataset.from_tensor_slices(image_path)
    dataset = dataset.map(load_and_preprocess_images, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(Config.BATCH_sIZE)
    return dataset

def get_train_and_valid_data():
    train_data = load_dataset(os.path.join(Config.DATA_PAth, 'train'))
    val_data = load_dataset(os.path.join(Config.DATA_PAth, 'val'))
    return train_data, val_data