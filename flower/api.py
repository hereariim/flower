from functools import wraps
import shutil
import tempfile

from aiohttp.web import HTTPBadRequest
from webargs import files, validate

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
    
    if kwargs["accept"]=="image/*":
        return open(filepath,"rb")

    