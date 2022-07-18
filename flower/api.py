from functools import wraps
import shutil
import tempfile
from marshmallow import missing

from tqdm import tqdm
from zipfile import ZipFile
from skimage.io import imread, imsave, imread_collection, concatenate_images
import numpy as np
import tensorflow as tf
from tensorflow import keras
from focal_loss import BinaryFocalLoss
from tensorflow.keras import backend as K
import numpy as np
from skimage.transform import resize

from PIL import Image
from aiohttp import web #HTTPBadRequest
from webargs import fields, validate
import matplotlib.pyplot as plt

import flower.config as cfg

import os
import sys
import random
import warnings

import numpy as np
import imageio
import cv2
import matplotlib.pyplot as plt


from tqdm import tqdm
from itertools import chain
from skimage.transform import resize
from skimage.morphology import label
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, Dense, Dropout, experimental
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from albumentations import Resize,PadIfNeeded

from albumentations import (VerticalFlip, HorizontalFlip, Flip, RandomScale, RandomRotate90, ShiftScaleRotate, CenterCrop, OpticalDistortion,                              
                            ElasticTransform, JpegCompression, HueSaturationValue, GridDistortion,
                            RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise, CLAHE, ChannelShuffle,
                            InvertImg, RandomGamma, ToGray, PadIfNeeded, RandomCrop, Compose, Transpose, RandomSizedCrop, OneOf, Normalize,PadIfNeeded,Resize)

import tensorflow as tf
import imshowpair
from PIL import Image
from collections import Counter
from focal_loss import BinaryFocalLoss

def _catch_error(f):
    @wraps(f)
    def wrap(*args,**kwargs):
        try:
            return f(*args,**kwargs)
        except Exception as e:
            raise web.HTTPBadRequest(reason=e)
    return wrap

def get_metadata():
    metadata = {
        "author":"xxx",
        "description":"xxx",
        "license":"MIT",
    }
    return metadata

def get_predict_args():
    """
    Input fields for the user (inference)
    """
    arg_dict = {
        "image": fields.Field(
            required=False,
            type="file",
            missing="None",
            location="form",
            description="Insert image or image set contained into compressed folder .zip",  # needed to be parsed by UI
        ),
        "accept": fields.Str(
            description="Media type(s) that is/are acceptable for the response.",
            missing='application/zip',
            validate=validate.OneOf(['application/zip', 'image/png', 'application/json']),
        ),
    }
    return arg_dict


@_catch_error
def predict(**kwargs):
    """
    OUTPUT
    """

    filepath = kwargs["image"].filename
    originalname = kwargs["image"].original_filename

    print("IMAGE")

    def redimension(image):
        X = np.zeros((1,256,256,3),dtype=np.uint8)
        img = imread(image)
        size_ = img.shape
        X[0] = resize(img, (256,256), mode="constant", preserve_range=True)
        return X,size_

    def dice_coefficient(y_true,y_pred):
        eps = 1e-6
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection =K.sum(y_true_f*y_pred_f)
        return (2. * intersection) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + eps)

    print("1")
    image_reshaped, size_ = redimension(filepath)
    x,y,z = size_
    print("2")
    model_new = tf.keras.models.load_model("best_model_FL_BCE_0_5.h5",custom_objects={"dice coefficient": dice_coefficient})
    print("3")
    prediction = model_new.predict(image_reshaped)
    print("4")
    preds_test_t = (prediction > 0.2)
    print("5")
    preds_test_t = resize(preds_test_t[0,:,:,0],(x,y),mode="constant",preserve_range=True)
    print("6")
    imsave(fname="demo.png", arr=np.squeeze(preds_test_t))
    print("SAVE")
    # data = imread(filepath)

 #   temp_dir = tempfile.TemporaryDirectory()
 #   with ZipFile(filepath,'r') as compressed_file:
 #       compressed_file.extractall(temp_dir.name)
 #       print('> Files extracted')

 #   image_sample = []
 #   for f in os.listdir(temp_dir.name):
 #       image_sample.append(os.path.join(temp_dir.name,f))

 #   print("> redimension")
 #   image_sample_reshaped = []
 #   size_original = []
 #   for i in tqdm(range(len(image_sample)),"en cours"):
 #       image_reshaped,size_ = redimension(image_sample[i])
 #       image_sample_reshaped.append(image_reshaped)
 #       x,y,z = size_
 #       size_original.append((x,y))
 #   print("done")
 #   print(tf.__version__)
 #   model_new = tf.keras.models.load_model("best_model_FL_BCE_0_5.h5",custom_objects={"dice_coefficient" : dice_coefficient})
 #    print("> model imported")

#    image_prediction = []
#    for sample in image_sample_reshaped:
#        prediction = model_new.predict(sample)
#        preds_test_t = (prediction > 0.2)
#        image_prediction.append(preds_test_t)
#    print("> prediction done")

#    image_output = []
#    for sample,i,name in zip(image_prediction,range(len(size_original)),image_sample):
#        preds_test_t = resize(preds_test_t[0,:,:,0], (x,y), mode = "constant", preserve_range = True)
#        imsave(fname=name+".png", arr=np.squeeze(preds_test_t))
        # plt.imsave('demo.png', BW_Original, cmap = plt.cm.gray)
#    print("SAVE")

    # Return the image directly
    if kwargs['accept'] == 'image/png':
        # img = Image.open(originalname)
        # return img.save("output")
        
        return open('demo.png','rb')
    
    # Return a zip
    elif kwargs['accept'] == 'application/zip':

        zip_dir = tempfile.TemporaryDirectory()

        # Add original image to output zip
        # for name in image_sample:
        #    shutil.copyfile(name+".png", zip_dir.name + '/'+ name+".png")
        shutil.copyfile("demo.png", zip_dir.name + "/demo.png")
        # Add for example a demo txt file
        with open(f'{zip_dir.name}/demo.txt', 'w') as f:
            f.write('Add here any additional information!')

        # Pack dir into zip and return it
        shutil.make_archive(
            zip_dir.name,
            format='zip',
            root_dir=zip_dir.name,
        )
        zip_path = zip_dir.name + '.zip'

        return open(zip_path, 'rb')
    