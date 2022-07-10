from functools import wraps
import shutil
import tempfile
from skimage.io import imread
from PIL import Image
from aiohttp import web #HTTPBadRequest
from webargs import fields, validate

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
    im = Image.open(filepath)
    im.save('demo.png')  
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
    