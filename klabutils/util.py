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

import klabutils
import klabutils.core

logger = logging.getLogger(__name__)
_not_importable = set()

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


def mkdir_exists_ok(path):
    try:
        os.makedirs(path)
        return True
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            return False
        else:
            raise