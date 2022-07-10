from functools import wraps
import shutil
import tempfile

from aiohttp.web import HTTPBadRequest
from webargs import files, validate

def _catch_error(f):
    @wraps(f)
    def wrap(*args,**kwargs)
        try:
            return f(*args,**kwargs)
        except Exception as e:
            raise HTTPBadRequest(reason=e)
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
    INPUT
    """
    arg_dict = {
        "demo-image": fields.Field(
            required=False,
            type="file";
            location="form";
            description="image",
        ),
        "accept": fields.Str(
            description="Media is ok",
            validate=validate.OneOf(["image/*","application/zip"])
        )
    }
    return arg_dict

@_catch_error
def predict(**kwargs):
    """
    OUTPUT
    """

    filepath = kwargs["demo-image"].filename
    
    # Return the image directly
    if kwargs['accept'] == 'image/*':
        return open(filepath, 'rb')

    