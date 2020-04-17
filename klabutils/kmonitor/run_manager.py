import errno
import json
import logging
import os
import re
import signal
import socket
import stat
import subprocess
import sys
import time
from tempfile import NamedTemporaryFile
import threading
import yaml
import numbers
import inspect
import glob
import platform
import fnmatch

from klabutils import Error
from klabutils.compat import windows

class LaunchError(Error):
    """Raised when there's an error starting up."""


class Process(object):
    """Represents a running process with an interface that
    mimics Popen's.

    Only works on Unix-y systems.

    TODO(adrian): probably rewrite using psutil.Process
    """

    def __init__(self, pid):
        self.returncode = None
        self.pid = pid

    def poll(self):
        if self.returncode is None:
            try:
                if platform.system() == "Windows":
                    if windows.pid_running(self.pid) == False:
                        raise OSError(0, "Process isn't running")
                else:
                    os.kill(self.pid, 0)
            except OSError as err:
                if err.errno == errno.ESRCH:
                    # ESRCH == No such process
                    # we have no way of getting the real return code, so just set it to 0
                    self.returncode = 0
                elif err.errno == errno.EPERM:
                    # EPERM clearly means there's a process to deny access to
                    pass
                else:
                    # According to "man 2 kill" possible error values are
                    # (EINVAL, EPERM, ESRCH)
                    raise

        return self.returncode

    def wait(self):
        while self.poll() is None:
            time.sleep(1)

    def interrupt(self):
        os.kill(self.pid, signal.SIGINT)

    def terminate(self):
        os.kill(self.pid, signal.SIGTERM)

    def kill(self):
        os.kill(self.pid, signal.SIGKILL)
