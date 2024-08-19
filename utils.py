# import tensorflow.keras.backend as K
# import tensorflow as tf
# from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np 
from scipy.ndimage import distance_transform_edt
import os 
from datetime import datetime
import logging
import torch
import cv2 as cv
import cv2
from PIL import Image, ImageDraw, ImageOps

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

"""
Create two logger named 'base' and 'val', with file name specified in 'phase' variable at the 
path specified in 'root' variable.
If 'screen=True', message is printed to screen
"""
def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False):
    '''set up logger'''
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        l.addHandler(sh)

def mkdir(dir_path):
    """
    Create a directory if it doesn't exist.

    Args:
        dir_path (str): Path of the directory to be created.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory '{dir_path}' created.")
    else:
        print(f"Directory '{dir_path}' already exists.")

def dice_coefficient(targets, inputs, axis=(0,1), smooth=1e-5):
    numerator = 2 * tf.reduce_sum(targets * inputs, axis=axis)
    denominator = tf.reduce_sum(targets + inputs, axis=axis) + smooth 
    return tf.reduce_mean(numerator / denominator)

def dice(targets, inputs, axis=(0,1), smooth=1e-5):
    numerator = 2 * np.sum(targets * inputs, axis=axis)
    denominator = np.sum(targets + inputs, axis=axis) + smooth 
    return np.mean(numerator / denominator)

def DiceLoss(y_true, y_pred):
    return 1-dice(y_true, y_pred)

def resolve_status(train_status):
    if train_status == 1:
        restore = False
        is_training = True
    if train_status == 2:
        restore = True
        is_training = True
    if train_status == 3:
        restore = True
        is_training = False

    return restore, is_training

def my_func(mask):
    epsilon = 0
    def bwdist(im): return distance_transform_edt(np.logical_not(im))
    bw = mask
    signed_dist = bwdist(bw) - bwdist(1 - bw)
    d = signed_dist.astype(np.float32)
    d += epsilon
    while np.count_nonzero(d < 0) < 5:
        d -= 1

    return d

def load_image(path, batch_size, label=False):
    if path.endswith('npy'):
        image = np.load(path)
    elif path.endswith('gif'):
        image = Image.open(path)
        image = np.asarray(image)
    elif path.endswith('tif'):
        image = Image.open(path)
        image = np.asarray(image)
#         print(image.min(), image.max)
    else: 
        image = cv2.imread(path, 0)
#     image = np.resize(image, (256, 256))
    image = image.astype('float32')
    if label:
#         image = image.astype('float32')
        image *= 1.0 / image.max()
#     else:
#         print(np.min(image), np.max(image))
#         image = (image - 0) * (255.0 / (np.max(image) - 0))
#         image = (image - np.min(image)) * (255.0 / (np.max(image) - np.min(image)))
#         image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image = np.asarray([image] * batch_size)
#     image = image[:, :, np.newaxis]

    return image

