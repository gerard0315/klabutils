from __future__ import absolute_import, print_function

__author__ = """Chris Van Pelt"""
__email__ = 'vanpelt@wandb.com'
__version__ = '0.8.31'

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
from six.moves import queue
from six import string_types
from importlib import import_module

from .. import env
from .. import io_wrap
from ..core import *

from klabutils import util
from klabutils import trigger
from klabutils import kmonitor
from klabutils.kmonitor import kmonitor_run
from klabutils.kmonitor import run_manager
from klabutils.kmonitor.run_manager import LaunchError, Process

_shutdown_async_log_thread_wait_time = 20

def test_monitor():
  print('kmonitor!')

run = None
# Stores what modules have been patched
patched = {
    "tensorboard": [],
    "keras": [],
    "gym": []
}
_saved_files = set()
_global_run_stack = []


def join(exit_code=None):
    """Marks a run as finished"""
    shutdown_async_log_thread()
    if run:
        run.close_files()
    if len(_global_run_stack) > 0:
        _global_run_stack.pop()


def save(glob_str, base_path=None, policy="live"):
    """ Ensure all files matching *glob_str* are synced to wandb with the policy specified.

    base_path: the base path to run the glob relative to
    policy:
        live: upload the file as it changes, overwriting the previous version
        end: only upload file when the run ends
    """
    global _saved_files
    if run is None:
        raise ValueError(
            "You must call `wandb.init` before calling save")
    if policy not in ("live", "end"):
        raise ValueError(
            'Only "live" and "end" policies are currently supported.')
    if isinstance(glob_str, bytes):
        glob_str = glob_str.decode('utf-8')
    if not isinstance(glob_str, string_types):
        raise ValueError("Must call wandb.save(glob_str) with glob_str a str")

    if base_path is None:
        base_path = os.path.dirname(glob_str)
    wandb_glob_str = os.path.relpath(glob_str, base_path)
    if "../" in wandb_glob_str:
        raise ValueError(
            "globs can't walk above base_path")
    if (glob_str, base_path, policy) in _saved_files:
        return []
    if glob_str.startswith("gs://") or glob_str.startswith("s3://"):
        termlog(
            "%s is a cloud storage url, can't save file to wandb." % glob_str)
        return []
    run.send_message(
        {"save_policy": {"glob": wandb_glob_str, "policy": policy}})
    files = []
    for path in glob.glob(glob_str):
        file_name = os.path.relpath(path, base_path)
        abs_path = os.path.abspath(path)
        wandb_path = os.path.join(run.dir, file_name)
        util.mkdir_exists_ok(os.path.dirname(wandb_path))
        # We overwrite existing symlinks because namespaces can change in Tensorboard
        if os.path.islink(wandb_path) and abs_path != os.readlink(wandb_path):
            os.remove(wandb_path)
            os.symlink(abs_path, wandb_path)
        elif not os.path.exists(wandb_path):
            os.symlink(abs_path, wandb_path)
        files.append(wandb_path)
    _saved_files.add((glob_str, base_path, policy))
    return files


_async_log_queue = queue.Queue()
_async_log_thread_shutdown_event = threading.Event()
_async_log_thread_complete_event = threading.Event()
_async_log_thread = None


def _async_log_thread_target():
    """Consumes async logs from our _async_log_queue and actually logs them"""
    global _async_log_thread
    shutdown_requested = False
    while not shutdown_requested:
        try:
            kwargs = _async_log_queue.get(block=True, timeout=1)
            log(**kwargs)
        except queue.Empty:
            shutdown_requested = _async_log_thread_shutdown_event.wait(1) and _async_log_queue.empty()
    _async_log_thread_complete_event.set()
    _async_log_thread = None


def _ensure_async_log_thread_started():
    """Ensures our log consuming thread is started"""
    global _async_log_thread, _async_log_thread_shutdown_event, _async_log_thread_complete_event

    if _async_log_thread is None:
        _async_log_thread_shutdown_event = threading.Event()
        _async_log_thread_complete_event = threading.Event()
        _async_log_thread = threading.Thread(target=_async_log_thread_target)
        _async_log_thread.daemon = True
        _async_log_thread.start()


def shutdown_async_log_thread():
    """Shuts down our async logging thread"""
    if _async_log_thread:
        _async_log_thread_shutdown_event.set()
        res = _async_log_thread_complete_event.wait(_shutdown_async_log_thread_wait_time)  # TODO: possible race here
        if res is False:
            termwarn('async log queue not empty after %d seconds, some log statements will be dropped' % (
                _shutdown_async_log_thread_wait_time))
            # FIXME: it is worse than this, likely the program will crash because files will be closed
        # FIXME: py 2.7 will return None here so we dont know if we dropped data


def log(row=None, commit=None, step=None, sync=True, *args, **kwargs):
    """Log a dict to the global run's history.

    wandb.log({'train-loss': 0.5, 'accuracy': 0.9})

    Args:
        row (dict, optional): A dict of serializable python objects i.e str: ints, floats, Tensors, dicts, or wandb.data_types
        commit (boolean, optional): Persist a set of metrics, if false just update the existing dict (defaults to true if step is not specified)
        step (integer, optional): The global step in processing. This persists any non-committed earlier steps but defaults to not committing the specified step
        sync (boolean, True): If set to False, process calls to log in a seperate thread
    """

    if run is None:
        raise ValueError(
            "You must call `wandb.init` in the same process before calling log")

    run.log(row, commit, step, sync, *args, **kwargs)


def reset_env(exclude=[]):
    """Remove environment variables, used in Jupyter notebooks"""
    if os.getenv(env.INITED):
        kmonitor_keys = [key for key in os.environ.keys() if key.startswith(
            'WANDB_') and key not in exclude]
        for key in kmonitor_keys:
            del os.environ[key]
        return True
    else:
        return False


def _get_python_type():
    try:
        if 'terminal' in get_ipython().__module__:
            return 'ipython'
        else:
            return 'jupyter'
    except (NameError, AttributeError):
        return "python"

def _init_jupyter(run):
    """Asks for user input to configure the machine if it isn't already and creates a new run.
    Log pushing and system stats don't start until `wandb.log()` is first called.
    """
    from klabutils.kmonitor import jupyter
    from IPython.core.display import display, HTML
    # TODO: Should we log to jupyter?
    # global logging had to be disabled because it set the level to debug
    # I also disabled run logging because we're rairly using it.
    # try_to_set_up_global_logging()
    # run.enable_logging()
    os.environ[env.JUPYTER] = "true"

    run.set_environment()
    run._init_jupyter_agent()
    ipython = get_ipython()
    ipython.register_magics(jupyter.WandBMagics)

    def reset_start():
        """Reset START_TIME to when the cell starts"""
        global START_TIME
        START_TIME = time.time()
    ipython.events.register("pre_run_cell", reset_start)

    def cleanup():
        # shutdown async logger because _user_process_finished isn't called in jupyter
        shutdown_async_log_thread()
        run._stop_jupyter_agent()
    ipython.events.register('post_run_cell', cleanup)


def init(job_type=None, dir=None, config=None, project=None, entity=None, reinit=None, tags=None,
         group=None, allow_val_change=False, resume=False, force=False, tensorboard=False,
         sync_tensorboard=False, monitor_gym=False, name=None, notes=None, id=None, magic=None,
         anonymous=None, config_exclude_keys=None, config_include_keys=None):
         
    init_args = locals()
    trigger.call('on_init', **init_args)
    global run
    global __stage_dir__
    global _global_watch_idx

    in_jupyter = _get_python_type() != "python"

    try:
        run = kmonitor_run.Run.from_environment_or_defaults()
        _global_run_stack.append(run)
    except IOError as e:
        termerror('Failed to create run directory: {}'.format(e))
        raise LaunchError("Could not write to filesystem.")

    run.set_environment()

    # if in_jupyter:
    #   _init_jupyter(run)
    # else:
    run.config.set_run_dir(run.dir)
    # set the run directory in the config so it actually gets persisted
    run.config.set_run_dir(run.dir)
    # we have re-read the config, add telemetry data

    # telemetry_updated = run.config._telemetry_update()

    run.history
    # Load the summary to support resuming
    run.summary.load()

    return run

keras = util.LazyLoader('keras', globals(), 'kmonitor.keras')


__all__ = ['init']
