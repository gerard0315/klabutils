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


def init(job_type=None, dir=None, config=None, project=None, entity=None, reinit=None, tags=None,
         group=None, allow_val_change=False, resume=False, force=False, tensorboard=False,
         sync_tensorboard=False, monitor_gym=False, name=None, notes=None, id=None, magic=None,
         anonymous=None, config_exclude_keys=None, config_include_keys=None):
    """Initialize W&B

    If called from within Jupyter, initializes a new run and waits for a call to
    `wandb.log` to begin pushing metrics.  Otherwise, spawns a new process
    to communicate with W&B.

    Args:
        job_type (str, optional): The type of job running, defaults to 'train'
        config (dict, argparse, or tf.FLAGS, optional): The hyper parameters to store with the run
        config_exclude_keys (list, optional): string keys to exclude storing in W&B when specifying config
        config_include_keys (list, optional): string keys to include storing in W&B when specifying config
        project (str, optional): The project to push metrics to
        entity (str, optional): The entity to push metrics to
        dir (str, optional): An absolute path to a directory where metadata will be stored
        group (str, optional): A unique string shared by all runs in a given group
        tags (list, optional): A list of tags to apply to the run
        id (str, optional): A globally unique (per project) identifier for the run
        name (str, optional): A display name which does not have to be unique
        notes (str, optional): A multiline string associated with the run
        reinit (bool, optional): Allow multiple calls to init in the same process
        resume (bool, str, optional): Automatically resume this run if run from the same machine,
            you can also pass a unique run_id
        sync_tensorboard (bool, optional): Synchronize wandb logs to tensorboard or tensorboardX
        force (bool, optional): Force authentication with wandb, defaults to False
        magic (bool, dict, or str, optional): magic configuration as bool, dict, json string,
            yaml filename
        anonymous (str, optional): Can be "allow", "must", or "never". Controls whether anonymous logging is allowed.
            Defaults to never.

    Returns:
        A wandb.run object for metric and config logging.
    """
    init_args = locals()
    trigger.call('on_init', **init_args)
    global run
    global __stage_dir__
    global _global_watch_idx

    # We allow re-initialization when we're in Jupyter or explicity opt-in to it.
    in_jupyter = _get_python_type() != "python"
    if reinit or (in_jupyter and reinit != False):
        # Reset global state for pytorch watch and tensorboard
        _global_watch_idx = 0
        if len(patched["tensorboard"]) > 0:
            util.get_module("wandb.tensorboard").reset_state()
        reset_env(exclude=env.immutable_keys())
        if len(_global_run_stack) > 0:
            if len(_global_run_stack) > 1:
                termwarn("If you want to track multiple runs concurrently in wandb you should use multi-processing not threads")
            join()
        run = None

    # TODO: deprecate tensorboard
    if tensorboard or sync_tensorboard and len(patched["tensorboard"]) == 0:
        util.get_module("wandb.tensorboard").patch()
    if monitor_gym and len(patched["gym"]) == 0:
        util.get_module("wandb.gym").monitor()

    sagemaker_config = util.parse_sm_config()
    tf_config = util.parse_tfjob_config()
    if group == None:
        group = os.getenv(env.RUN_GROUP)
    if job_type == None:
        job_type = os.getenv(env.JOB_TYPE)
    if sagemaker_config:
        # Set run_id and potentially grouping if we're in SageMaker
        run_id = os.getenv('TRAINING_JOB_NAME')
        if run_id:
            os.environ[env.RUN_ID] = '-'.join([
                run_id,
                os.getenv('CURRENT_HOST', socket.gethostname())])
        conf = json.load(
            open("/opt/ml/input/config/resourceconfig.json"))
        if group == None and len(conf["hosts"]) > 1:
            group = os.getenv('TRAINING_JOB_NAME')
        # Set secret variables
        if os.path.exists("secrets.env"):
            for line in open("secrets.env", "r"):
                key, val = line.strip().split('=', 1)
                os.environ[key] = val
    elif tf_config:
        cluster = tf_config.get('cluster')
        job_name = tf_config.get('task', {}).get('type')
        task_index = tf_config.get('task', {}).get('index')
        if job_name is not None and task_index is not None:
            # TODO: set run_id for resuming?
            run_id = cluster[job_name][task_index].rsplit(":")[0]
            if job_type == None:
                job_type = job_name
            if group == None and len(cluster.get("worker", [])) > 0:
                group = cluster[job_name][0].rsplit("-"+job_name, 1)[0]
    image = util.image_id_from_k8s()
    if image:
        os.environ[env.DOCKER] = image

    if not os.environ.get(env.SWEEP_ID):
        if project:
            os.environ[env.PROJECT] = project
        if entity:
            os.environ[env.ENTITY] = entity
    else:
        if entity and entity != os.environ.get(env.ENTITY):
            termwarn("Ignoring entity='{}' passed to wandb.init when running a sweep".format(entity))
        if project and project != os.environ.get(env.PROJECT):
            termwarn("Ignoring project='{}' passed to wandb.init when running a sweep".format(project))

    if group:
        os.environ[env.RUN_GROUP] = group
    if job_type:
        os.environ[env.JOB_TYPE] = job_type
    if tags:
        if isinstance(tags, str):
            # People sometimes pass a string instead of an array of strings...
            tags = [tags]
        os.environ[env.TAGS] = ",".join(tags)
    if id:
        os.environ[env.RUN_ID] = id
        if name is None and resume is not "must":
            # We do this because of https://github.com/wandb/core/issues/2170
            # to ensure that the run's name is explicitly set to match its
            # id. If we don't do this and the id is eight characters long, the
            # backend will set the name to a generated human-friendly value.
            #
            # In any case, if the user is explicitly setting `id` but not
            # `name`, their id is probably a meaningful string that we can
            # use to label the run.
            #
            # In the resume="must" case, we know we are resuming, so we should
            # make sure to not set the name because it would have been set with
            # the original run.
            #
            # TODO: handle "auto" resume by moving this logic later when we know
            # if there is a resume.
            name = os.environ.get(env.NAME, id)  # environment variable takes precedence over this.
    if name:
        os.environ[env.NAME] = name
    if notes:
        os.environ[env.NOTES] = notes
    if magic is not None and magic is not False:
        if isinstance(magic, dict):
            os.environ[env.MAGIC] = json.dumps(magic)
        elif isinstance(magic, str):
            os.environ[env.MAGIC] = magic
        elif isinstance(magic, bool):
            pass
        else:
            termwarn("wandb.init called with invalid magic parameter type", repeat=False)
        from wandb import magic_impl
        magic_impl.magic_install(init_args=init_args)
    if dir:
        os.environ[env.DIR] = dir
        util.mkdir_exists_ok(wandb_dir())
    if anonymous is not None:
        os.environ[env.ANONYMOUS] = anonymous
    if os.environ.get(env.ANONYMOUS, "never") not in ["allow", "must", "never"]:
        raise LaunchError("anonymous must be set to 'allow', 'must', or 'never'")

    resume_path = os.path.join(wandb_dir(), wandb_run.RESUME_FNAME)
    if resume == True:
        os.environ[env.RESUME] = "auto"
    elif resume in ("allow", "must", "never"):
        os.environ[env.RESUME] = resume
        if id:
            os.environ[env.RUN_ID] = id
    elif resume:
        os.environ[env.RESUME] = os.environ.get(env.RESUME, "allow")
        # TODO: remove allowing resume as a string in the future
        os.environ[env.RUN_ID] = id or resume
    elif os.path.exists(resume_path):
        os.remove(resume_path)
    if os.environ.get(env.RESUME) == 'auto' and os.path.exists(resume_path):
        if not os.environ.get(env.RUN_ID):
            os.environ[env.RUN_ID] = json.load(open(resume_path))["run_id"]

    # the following line is useful to ensure that no W&B logging happens in the user
    # process that might interfere with what they do
    # logging.basicConfig(format='user process %(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # If a thread calls wandb.init() it will get the same Run object as
    # the parent. If a child process with distinct memory space calls
    # wandb.init(), it won't get an error, but it will get a result of
    # None.
    # This check ensures that a child process can safely call wandb.init()
    # after a parent has (only the parent will create the Run object).
    # This doesn't protect against the case where the parent doesn't call
    # wandb.init but two children do.
    if run or os.getenv(env.INITED):
        return run

    if __stage_dir__ is None:
        __stage_dir__ = "wandb"
        util.mkdir_exists_ok(wandb_dir())

    try:
        signal.signal(signal.SIGQUIT, _debugger)
    except AttributeError:
        pass

    try:
        run = wandb_run.Run.from_environment_or_defaults()
        _global_run_stack.append(run)
    except IOError as e:
        termerror('Failed to create run directory: {}'.format(e))
        raise LaunchError("Could not write to filesystem.")

    run.set_environment()

    def set_global_config(run):
        global config  # because we already have a local config
        config = run.config
    set_global_config(run)
    global summary
    summary = run.summary

    # set this immediately after setting the run and the config. if there is an
    # exception after this it'll probably break the user script anyway
    os.environ[env.INITED] = '1'

    if in_jupyter:
        _init_jupyter(run)
    elif run.mode == 'clirun':
        pass
    elif run.mode == 'run':
        api = InternalApi()
        # let init_jupyter handle this itself
        if not in_jupyter and not api.api_key:
            termlog(
                "W&B is a tool that helps track and visualize machine learning experiments")
            if force:
                termerror(
                    "No credentials found.  Run \"wandb login\" or \"wandb off\" to disable wandb")
            else:
                if util.prompt_api_key(api):
                    _init_headless(run)
                else:
                    termlog(
                        "No credentials found.  Run \"wandb login\" to visualize your metrics")
                    run.mode = "dryrun"
                    _init_headless(run, False)
        else:
            _init_headless(run)
    elif run.mode == 'dryrun':
        termlog(
            'Dry run mode, not syncing to the cloud.')
        _init_headless(run, False)
    else:
        termerror(
            'Invalid run mode "%s". Please unset WANDB_MODE.' % run.mode)
        raise LaunchError("The WANDB_MODE environment variable is invalid.")

    # set the run directory in the config so it actually gets persisted
    run.config.set_run_dir(run.dir)
    # we have re-read the config, add telemetry data
    telemetry_updated = run.config._telemetry_update()

    if sagemaker_config:
        run.config._update(sagemaker_config)
        allow_val_change = True
    if config or telemetry_updated:
        run.config._update(config,
                exclude_keys=config_exclude_keys,
                include_keys=config_include_keys,
                allow_val_change=allow_val_change,
                as_defaults=not allow_val_change)

    # Access history to ensure resumed is set when resuming
    run.history
    # Load the summary to support resuming
    run.summary.load()

    return run


__all__ = ['init', 'config', 'summary', 'join', 'log', 'save', 'restore',
    'tensorflow', 'watch', 'types', 'tensorboard', 'jupyter', 'keras', 'fastai',
    'docker', 'xgboost', 'gym', 'ray', 'run', 'join', 'Image', 'Video',
    'Audio',  'Table', 'Html', 'Object3D', 'Molecule', 'Histogram', 'Graph', 'Api']