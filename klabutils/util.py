from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import base64
import colorsys
import errno
import hashlib
import json
import getpass
import logging
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import random
import platform
import stat
import shortuuid
import importlib
import types
import yaml
import numbers
from datetime import date, datetime

import click
import requests
import six
from six.moves import queue
import textwrap
from sys import getsizeof
from collections import namedtuple
from importlib import import_module

from .core import *

import klabutils

logger = logging.getLogger(__name__)
_not_importable = set()


def generate_id():
    # ~3t run ids (36**8)
    run_gen = shortuuid.ShortUUID(alphabet=list(
        "0123456789abcdefghijklmnopqrstuvwxyz"))
    return run_gen.random(8)


def get_module(name, required=None):
    """
    Return module or None. Absolute import is required.
    :param (str) name: Dot-separated module path. E.g., 'scipy.stats'.
    :param (str) required: A string to raise a ValueError if missing
    :return: (module|None) If import succeeds, the module will be returned.
    """
    if name not in _not_importable:
        try:
            return import_module(name)
        except Exception as e:
            _not_importable.add(name)
            msg = "Error importing optional module {}".format(name)
            if required:
                logger.exception(msg)
    if required and name in _not_importable:
        raise klabutils.Error(required)


def parse_tfjob_config():
    """Attempts to parse TFJob config, returning False if it can't find it"""
    if os.getenv("TF_CONFIG"):
        try:
            return json.loads(os.environ["TF_CONFIG"])
        except ValueError:
            return False
    else:
        return False


def parse_sm_config():
    """Attempts to parse SageMaker configuration returning False if it can't find it"""
    sagemaker_config = "/opt/ml/input/config/hyperparameters.json"
    resource_config = "/opt/ml/input/config/resourceconfig.json"
    if os.path.exists(sagemaker_config) and os.path.exists(resource_config):
        conf = {}
        conf["sagemaker_training_job_name"] = os.getenv('TRAINING_JOB_NAME')
        # Hyper-parameter searchs quote configs...
        for k, v in six.iteritems(json.load(open(sagemaker_config))):
            cast = v.strip('"')
            if os.getenv("WANDB_API_KEY") is None and k == "wandb_api_key":
                os.environ["WANDB_API_KEY"] = cast
            else:
                if re.match(r'^[-\d]+$', cast):
                    cast = int(cast)
                elif re.match(r'^[-.\d]+$', cast):
                    cast = float(cast)
                conf[k] = cast
        return conf
    else:
        return False


class KJSONEncoder(json.JSONEncoder):
    """A JSON Encoder that handles some extra types."""

    def default(self, obj):
        tmp_obj, converted = json_friendly(obj)
        tmp_obj, compressed = maybe_compress_summary(
            tmp_obj, get_h5_typename(obj))
        if converted:
            return tmp_obj
        return json.JSONEncoder.default(self, tmp_obj)


class KHistoryJSONEncoder(json.JSONEncoder):
    """A JSON Encoder that handles some extra types.
    This encoder turns numpy like objects with a size > 32 into histograms"""

    def default(self, obj):
        obj, converted = json_friendly(obj)
        obj, compressed = maybe_compress_history(obj)
        if converted:
            return obj
        return json.JSONEncoder.default(self, obj)

class JSONEncoderUncompressed(json.JSONEncoder):
    """A JSON Encoder that handles some extra types.
    This encoder turns numpy like objects with a size > 32 into histograms"""

    def default(self, obj):
        if is_numpy_array(obj):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def class_colors(class_count):
    # make class 0 black, and the rest equally spaced fully saturated hues
    return [[0, 0, 0]] + [colorsys.hsv_to_rgb(i / (class_count - 1.), 1.0, 1.0) for i in range(class_count-1)]

def json_dump_safer(obj, fp, **kwargs):
    """Convert obj to json, with some extra encodable types."""
    return json.dump(obj, fp, cls=KJSONEncoder, **kwargs)

def json_dumps_safer(obj, **kwargs):
    """Convert obj to json, with some extra encodable types."""
    return json.dumps(obj, cls=KJSONEncoder, **kwargs)

# This is used for dumping raw json into files
def json_dump_uncompressed(obj, fp, **kwargs):
    """Convert obj to json, with some extra encodable types."""
    return json.dump(obj, fp, cls=JSONEncoderUncompressed, **kwargs)

def json_dumps_safer_history(obj, **kwargs):
    """Convert obj to json, with some extra encodable types, including histograms"""
    return json.dumps(obj, cls=KHistoryJSONEncoder, **kwargs)

def make_json_if_not_number(v):
    """If v is not a basic type convert it to json."""
    if isinstance(v, (float, int)):
        return v
    return json_dumps_safer(v)


def mkdir_exists_ok(path):
    try:
        os.makedirs(path)
        return True
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            return False
        else:
            raise


def guess_data_type(shape, risky=False):
    """Infer the type of data based on the shape of the tensors

    Args:
        risky(bool): some guesses are more likely to be wrong.
    """
    # (samples,) or (samples,logits)
    if len(shape) in (1, 2):
        return 'label'
    # Assume image mask like fashion mnist: (no color channel)
    # This is risky because RNNs often have 3 dim tensors: batch, time, channels
    if risky and len(shape) == 3:
        return 'image'
    if len(shape) == 4:
        if shape[-1] in (1, 3, 4):
            # (samples, height, width, Y \ RGB \ RGBA)
            return 'image'
        else:
            # (samples, height, width, logits)
            return 'segmentation_mask'
    return None


def has_num(dictionary, key):
     return (key in dictionary and isinstance(dictionary[key], numbers.Number))


np = get_module('numpy')

MAX_SLEEP_SECONDS = 60 * 5
# TODO: Revisit these limits
VALUE_BYTES_LIMIT = 100000


def get_full_typename(o):
    """We determine types based on type names so we don't have to import
    (and therefore depend on) PyTorch, TensorFlow, etc.
    """
    instance_name = o.__class__.__module__ + "." + o.__class__.__name__
    if instance_name in ["builtins.module", "__builtin__.module"]:
        return o.__name__
    else:
        return instance_name


def get_h5_typename(o):
    typename = get_full_typename(o)
    if is_tf_tensor_typename(typename):
        return "tensorflow.Tensor"
    elif is_pytorch_tensor_typename(typename):
        return "torch.Tensor"
    else:
        return o.__class__.__module__.split('.')[0] + "." + o.__class__.__name__


def is_tf_tensor(obj):
    import tensorflow
    return isinstance(obj, tensorflow.Tensor)


def is_tf_tensor_typename(typename):
    return typename.startswith('tensorflow.') and ('Tensor' in typename or 'Variable' in typename)


def is_tf_eager_tensor_typename(typename):
    return typename.startswith('tensorflow.') and ('EagerTensor' in typename)


def is_pytorch_tensor(obj):
    import torch
    return isinstance(obj, torch.Tensor)


def is_pytorch_tensor_typename(typename):
    return typename.startswith('torch.') and ('Tensor' in typename or 'Variable' in typename)


def is_pandas_data_frame_typename(typename):
    return typename.startswith('pandas.') and 'DataFrame' in typename


def is_matplotlib_typename(typename):
    return typename.startswith("matplotlib.")


def is_plotly_typename(typename):
    return typename.startswith("plotly.")


def is_plotly_figure_typename(typename):
    return typename.startswith("plotly.") and typename.endswith('.Figure')


def is_numpy_array(obj):
    return np and isinstance(obj, np.ndarray)


def is_pandas_data_frame(obj):
    return is_pandas_data_frame_typename(get_full_typename(obj))


def ensure_matplotlib_figure(obj):
    """Extract the current figure from a matplotlib object or return the object if it's a figure.
    raises ValueError if the object can't be converted.
    """
    import matplotlib
    from matplotlib.figure import Figure
    if obj == matplotlib.pyplot:
        obj = obj.gcf()
    elif not isinstance(obj, Figure):
        if hasattr(obj, "figure"):
            obj = obj.figure
            # Some matplotlib objects have a figure function
            if not isinstance(obj, Figure):
                raise ValueError(
                    "Only matplotlib.pyplot or matplotlib.pyplot.Figure objects are accepted.")
    if not obj.gca().has_data():
        raise ValueError(
            "You attempted to log an empty plot, pass a figure directly or ensure the global plot isn't closed.")
    return obj


def json_friendly(obj):
    """Convert an object into something that's more becoming of JSON"""
    converted = True
    typename = get_full_typename(obj)

    if is_tf_eager_tensor_typename(typename):
        obj = obj.numpy()
    elif is_tf_tensor_typename(typename):
        obj = obj.eval()
    elif is_pytorch_tensor_typename(typename):
        try:
            if obj.requires_grad:
                obj = obj.detach()
        except AttributeError:
            pass  # before 0.4 is only present on variables

        try:
            obj = obj.data
        except RuntimeError:
            pass  # happens for Tensors before 0.4

        if obj.size():
            obj = obj.numpy()
        else:
            return obj.item(), True

    if is_numpy_array(obj):
        if obj.size == 1:
            obj = obj.flatten()[0]
        elif obj.size <= 32:
            obj = obj.tolist()
    elif np and isinstance(obj, np.generic):
        obj = obj.item()
    elif isinstance(obj, bytes):
        obj = obj.decode('utf-8')
    elif isinstance(obj, (datetime, date)):
        obj = obj.isoformat()
    else:
        converted = False
    if getsizeof(obj) > VALUE_BYTES_LIMIT:
        klabutils.termwarn("Serializing object of type {} that is {} bytes".format(type(obj).__name__, getsizeof(obj)))

    return obj, converted
        

def to_forward_slash_path(path):
    if platform.system() == "Windows":
        path = path.replace("\\", "/")
    return path


def maybe_compress_history(obj):
    if np and isinstance(obj, np.ndarray) and obj.size > 32:
        return klabutils.Histogram(obj, num_bins=32).to_json(), True
    else:
        return obj, False


def maybe_compress_summary(obj, h5_typename):
    if np and isinstance(obj, np.ndarray) and obj.size > 32:
        return {
            "_type": h5_typename,  # may not be ndarray
            "var": np.var(obj).item(),
            "mean": np.mean(obj).item(),
            "min": np.amin(obj).item(),
            "max": np.amax(obj).item(),
            "10%": np.percentile(obj, 10),
            "25%": np.percentile(obj, 25),
            "75%": np.percentile(obj, 75),
            "90%": np.percentile(obj, 90),
            "size": obj.size
        }, True
    else:
        return obj, False


def image_id_from_k8s():
    """Pings the k8s metadata service for the image id"""
    token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
    if os.path.exists(token_path):
        k8s_server = "https://{}:{}/api/v1/namespaces/default/pods/{}".format(
            os.getenv("KUBERNETES_SERVICE_HOST"), os.getenv(
                "KUBERNETES_PORT_443_TCP_PORT"), os.getenv("HOSTNAME")
        )
        try:
            res = requests.get(k8s_server, verify="/var/run/secrets/kubernetes.io/serviceaccount/ca.crt",
                               timeout=3, headers={"Authorization": "Bearer {}".format(open(token_path).read())})
            res.raise_for_status()
        except requests.RequestException:
            return None
        try:
            return res.json()["status"]["containerStatuses"][0]["imageID"].strip("docker-pullable://")
        except (ValueError, KeyError, IndexError):
            logger.exception("Error checking kubernetes for image id")
            return None
            