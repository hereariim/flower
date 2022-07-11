from functools import wraps
import shutil
import tempfile

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
    Input fields for the user.
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

    print("IMAGE")

    def redimension(image):
        X = np.zeors((1,256,256,3),dtype=np.uint8)
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
    