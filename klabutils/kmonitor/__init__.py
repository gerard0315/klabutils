from __future__ import absolute_import, print_function

import atexit
import click
import io
import json
import logging
import time
import os
import contextlib
import signal
import six
import getpass
import socket
import subprocess
import sys
import traceback
import tempfile
import re
import glob
import threading
import platform
import collections
from importlib import import_module

from ..core import *

def test_monitor():
  print('kmonitor!')
