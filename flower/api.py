from functools import wraps
import shutil
import tempfile
from marshmallow import missing

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

def get_train_args():
    """
    Input fileds for the user (training)
    HYPERPARAMETER
    """
    arg_dict = {
        "learning rate" : fields.Str(
            required=False,
            missing="0.0007",
            description="Learning rate",
        ),
        "loss function" : fields.Str(
            missing="Weighted BCE",
            enum = ["Weighted BCE","Focal loss"],
            description = "Choose your loss function",
            required=False,
        ),
        "image augmentation" : fields.Str(
            missing=False,
            enum = ["Yes","No"],
            description = "Applying image augmentation",
            required=False,
        ),
        "batch size" : fields.Str(
            required=False,
            missing="2",
            description="Batch size",
        ),
        "filters" : fields.Str(
            required=False,
            missing="3",
            description="N filters",
        ),
        "accept": fields.Str(
            description="Media type(s) that is/are acceptable for the response.",
            missing='application/zip',
            validate=validate.OneOf(['application/zip', 'image/png', 'application/json']),
        ),
    }
    return arg_dict

@_catch_error
def train(**kwargs):
    """
    OUTPUT
    """

    results = {
        "status" : "ok",
        "train_args" : {},
        "dice" : {},
    }

    results["train_args"] = kwargs # input utilisateur

    #Input
    archive = cfg.DATA_DIR

    def fabriquer_train(archive):
        dico = {}
        for i in archive.namelist():
            i_temp = i.split("/")
            if i_temp[-1]!='':
                if i_temp[1]=='images':
                    imgfile = archive.open(i)
                    img = Image.open(imgfile)
                    dico[i_temp[-1]]=np.array(img)
        return dico

    def fabriquer_test(archive):
        dico = {}
        for i in archive.namelist():
            i_temp = i.split("/")
            if i_temp[-1]!='':
                if i_temp[1]=='masks':
                    imgfile = archive.open(i)
                    img = Image.open(imgfile)
                    dico[i_temp[-1]]=np.array(img)
        return dico

    image_ = fabriquer_train(archive)
    masks_ = fabriquer_test(archive)
    print("Input done")

    image_name = list(image_.keys())
    masks_name = list(masks_.keys())

    #Data
    from sklearn.model_selection import train_test_split
    X_train, X_test_, Y_train, Y_test_ = train_test_split(image_name, masks_name, test_size=0.2, random_state=42)
    x_train_, x_val_, y_train_, y_val_ = train_test_split(X_train, Y_train, test_size=0.2, random_state=42) #text_size=0.3 (à moi, 0.2)

    # Set some parameters
    IMG_WIDTH = 256
    IMG_HEIGHT = 256
    IMG_CHANNELS = 3

    warnings.filterwarnings('ignore', category=UserWarning, module='skimage')


    def get_X_data(ids):
        ids.sort()
        X = np.zeros((len(ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
        for n, id_ in tqdm(enumerate(ids), total=len(ids)):
            # we'll be using skimage library for reading file
            img = image_[id_][:,:,:IMG_CHANNELS]
            img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
            X[n] = img
        return X

    def get_Y_data(ids):
        ids.sort()
        Y = np.zeros((len(ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        
        for n, id_ in tqdm(enumerate(ids), total=len(ids)):
            mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
            masque_ = masks_[id_]
            gray_file = rgb2gray(masque_)
            threshold = threshold_otsu(gray_file)
            binary_file = (gray_file > threshold)
            masque_ = np.expand_dims(resize(binary_file, (IMG_HEIGHT, IMG_WIDTH), mode='constant',preserve_range=True), axis=-1)
            Y[n] = masque_
        return Y

    print('Getting and resizing train images ... ')

    x_train = get_X_data(x_train_)
    y_train = get_Y_data(y_train_)

    x_val = get_X_data(x_val_)
    y_val = get_Y_data(y_val_)

    X_test = get_X_data(X_test_)
    Y_test = get_Y_data(Y_test_)

    #Unet structure
    def conv2d_block(input_tensor, n_filters, kernel_size=3):
        # first layer
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
                padding="same")(input_tensor) # padding="valid"
        x = Activation("relu")(x)
        # second layer
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), 
                padding="same")(x)
        x = Activation("relu")(x)
        return x   

    def get_unet(input_img, n_filters,kernel_size=3):
        # contracting path # encoder
        c1 = conv2d_block(input_img, n_filters=n_filters*4, kernel_size=3) #The first block of U-net
        p1 = MaxPooling2D((2, 2)) (c1)

        c2 = conv2d_block(p1, n_filters=n_filters*8, kernel_size=3)
        p2 = MaxPooling2D((2, 2)) (c2)

        c3 = conv2d_block(p2, n_filters=n_filters*16, kernel_size=3)
        p3 = MaxPooling2D((2, 2)) (c3)

        c10 = conv2d_block(p3, n_filters=n_filters*16, kernel_size=3)
        p10 = MaxPooling2D((2, 2)) (c10)

        c12 = conv2d_block(p10, n_filters=n_filters*16, kernel_size=3)
        p12 = MaxPooling2D((2, 2)) (c12)

        c14 = conv2d_block(p12, n_filters=n_filters*16, kernel_size=3)
        p14 = MaxPooling2D((2, 2)) (c14)

        c4 = conv2d_block(p14, n_filters=n_filters*32, kernel_size=3)
        p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
        
        c5 = conv2d_block(p4, n_filters=n_filters*64, kernel_size=3) # last layer on encoding path 
        
        # expansive path # decoder
        u6 = Conv2DTranspose(n_filters*32, (3, 3), strides=(2, 2), padding='same') (c5) #upsampling included
        u6 = concatenate([u6, c4])
        c6 = conv2d_block(u6, n_filters=n_filters*32, kernel_size=3)

        u15 = Conv2DTranspose(n_filters*16, (3, 3), strides=(2, 2), padding='same') (c6)
        u15 = concatenate([u15, c14])
        c15 = conv2d_block(u15, n_filters=n_filters*16, kernel_size=3)

        u13 = Conv2DTranspose(n_filters*16, (3, 3), strides=(2, 2), padding='same') (c15)
        u13 = concatenate([u13, c12])
        c13 = conv2d_block(u13, n_filters=n_filters*16, kernel_size=3)

        u11 = Conv2DTranspose(n_filters*16, (3, 3), strides=(2, 2), padding='same') (c13)
        u11 = concatenate([u11, c10])
        c11 = conv2d_block(u11, n_filters=n_filters*16, kernel_size=3)

        u7 = Conv2DTranspose(n_filters*16, (3, 3), strides=(2, 2), padding='same') (c11)
        u7 = concatenate([u7, c3])
        c7 = conv2d_block(u7, n_filters=n_filters*16, kernel_size=3)

        u8 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c7)
        u8 = concatenate([u8, c2])
        c8 = conv2d_block(u8, n_filters=n_filters*8, kernel_size=3)

        u9 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = conv2d_block(u9, n_filters=n_filters*4, kernel_size=3)
        
        outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
        model = Model(inputs=[input_img], outputs=[outputs])
        return model

    #Dice coefficient
    def dice_coefficient(y_true, y_pred):
        eps = 1e-6
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + eps) #eps pour éviter la division par 0 

    # Initialize model
    input_img = Input((256,256, 3), name='img')
    model = get_unet(input_img, n_filters=2, kernel_size=3) #nombre de filtre
    ###
    opt = tf.keras.optimizers.Adam(learning_rate=0.007)
    model.compile(optimizer=opt, loss=[BinaryFocalLoss(gamma=0.5)], metrics=[dice_coefficient]) #focal loss
    model.save_weights("best_model_FL_BCE_0_5.h5")
    model.summary()
    ###
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, verbose=1, mode='auto') #Stop training when a monitored metric has stopped improving.

    checkpoint_filepath = 'bestmodel.h5'
    Model_check = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True,
        save_weights_only=False, mode='auto') #Callback to save the Keras model or model weights at some frequency.
    ###
    results = model.fit(x_train,y_train, 
                    validation_data=(x_val,y_val),
                    epochs=50, batch_size = 3,
                    callbacks=[early_stop,Model_check])

    model_New = tf.keras.models.load_model('bestmodel.h5',custom_objects={'dice_coefficient': dice_coefficient})

    #step 3

    Mask_valid_pred_int= model_New.predict(x_val, verbose=2)

    from sklearn.metrics import f1_score

    # compute F1-score for a set of thresholds from (0.1 to 0.9 with a step of 0.1)
    prob_thresh = [i*10**-1 for i in range(1,10)]
    perf=[] # define an empty array to store the computed F1-score for each threshold
    perf_ALL=[]
    for r in tqdm(prob_thresh): # all th thrshold values
        preds_bin = ((Mask_valid_pred_int> r) + 0 )
        preds_bin1=preds_bin[:,:,:,0]
        GTALL=y_val[:,:,:,0]
        for ii in range(len(GTALL)): # all validation images
            predmask=preds_bin1[ii,:,:]
            GT=GTALL[ii,:,:]
            l = GT.flatten()
            p= predmask.flatten()
            perf.append(f1_score(l, p)) # re invert the maps: cells: 1, back :0
        perf_ALL.append(np.mean(perf))
        perf=[]

    max_f1 = max(perf_ALL)  # Find the maximum y value
    op_thr = prob_thresh[np.array(perf_ALL).argmax()]  # Find the x value corresponding to the maximum y value

    preds_test = model_New.predict(X_test, verbose=1)
    # we apply a threshold on predicted mask (probability mask) to convert it to a binary mask.
    preds_test_opt = (preds_test >op_thr).astype(np.uint8)

    PIXEL_TEST = []
    PIXEL_PRED = []
    for ix in range(len(X_test_)):
    a = Y_test[ix, :, :, 0]
    b = preds_test_opt[ix, :, :, 0]
    for i in range(256):
        for j in range(256):
        PIXEL_TEST.append(int(a[i][j]))
        PIXEL_PRED.append(int(b[i][j]))
    
    Y_t = K.constant(PIXEL_TEST)
    pred_t = K.constant(PIXEL_PRED)
    dice = K.get_value(dice_coefficient(Y_t,pred_t))
    m = tf.keras.metrics.MeanIoU(num_classes=2)
    m.update_state(Y_t,pred_t)
    jaccard = m.result().numpy()

    results["dice"] = {"dice":dice}

    return results


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
            description="image",  # needed to be parsed by UI
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

    print(kwargs["image"])

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

    #data = imread(filepath)
    image_reshaped , size_ = redimension(filepath)
    print("> redimension")
    x,y,z = size_
    model_new = tf.keras.models.load_model("best_model_FL_BCE_0_5.h5",custom_objects={"dice_coefficient" : dice_coefficient})
    print("> model imported")
    prediction = model_new.predict(image_reshaped)
    print("> model predict")
    preds_test_t = (prediction > 0.2)
    print("> threshold optimization")
    preds_test_t = resize(preds_test_t[0,:,:,0], (x,y), mode = "constant", preserve_range = True)
    print("> resize done")
    imsave(fname="demo.png", arr=np.squeeze(preds_test_t))
    #plt.imsave('demo.png', BW_Original, cmap = plt.cm.gray)
    print("SAVE")

    # Return the image directly
    if kwargs['accept'] == 'image/png':
        #img = Image.open(originalname)
        #return img.save("output")
        
        return open('demo.png','rb')
    
    # Return a zip
    elif kwargs['accept'] == 'application/zip':

        zip_dir = tempfile.TemporaryDirectory()

        # Add original image to output zip
        shutil.copyfile('demo.png',
                        zip_dir.name + '/image.png')

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
    