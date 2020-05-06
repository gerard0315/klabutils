import datetime
import logging
import os
import socket
import json
import yaml
import fnmatch
import tempfile
import shutil
import glob
import collections

from .. import env
from .. import io_wrap
from ..core import *

from klabutils import util

import six
from six.moves import input
from six.moves import urllib
import atexit
import sys

# metric keys
STEP_ = 'step'
LOSS_ = 'loss'
ACC_ = 'acc'
TIMESTAMP_ = 'timestamp'

# attribute?
BEST_EPOCH = 'best_epoch'
LOGS_DEFAULT = ''
LOGS_VAL = 'val_'
LOGS_TRAIN = 'train_'

# environmental
PROJECT_ID = 'PROJECT_ID'
MODEL_VERSION = 'MODEL_VERSION'
EXPERIMENT_ID = 'EXPERIMENT_ID' 

class Monitor:
  '''
    member variable用camelCase, function用underscore
  ''' 
  def __init__(self):
    # meta are all from injected env variables, nothing to do with the untrained model
    self._init_project_meta()

    self._logs = {LOGS_DEFAULT: [], LOGS_TRAIN: [], LOGS_VAL: []}
    self._concluded = False
    self._results = {}
    self.best_epoch = -1 # no idea about the definition of best_epoch
    self.best_acc = 0


  def log_step(self, step=-1, loss=999, acc=0, prefix=LOGS_DEFAULT):
    self._logs[prefix].append({STEP_: step, prefix+LOSS_: loss, prefix+ACC_: acc, TIMESTAMP_: util.nano_time()})

  def upload_logs(self):
    # upload to s3
    pass

  def conclude(self, show_logs=False):
    if not self._concluded:
      for k in self._logs.keys():
        self._logs[k] = sorted(self._logs[k], key=lambda a: a[TIMESTAMP_])
    if show_logs:
      print(self._logs)

  def _init_project_meta(self):
    # env
    self._projectID = os.getenv(PROJECT_ID)
    self._experimentID = os.getenv(EXPERIMENT_ID)
    self._modelVersion = os.getenv(MODEL_VERSION)
    self._defaultPath = '/home/kesci/work/{}_{}_{}.json'.format(self._projectID, self._modelVersion, self._experimentID)
 
  def conclude_and_save_to_file(self, path='', show_logs=False):
    self.conclude(show_logs)
    if not path:
      path = self._defaultPath
    with open(path, 'w') as f:
      f.write(self._json_object())

  def _json_object(self):
    obj = {
      'logs': self._logs,
      'projectID': self._projectID,
      'experimentID': self._experimentID, 
      'modelVersion': self._modelVersion, 
    }
    return json.dumps(obj)