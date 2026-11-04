import os
import sys
import tempfile
import time
import unittest

import psutil

from dgenerate.console.procmon import ProcessMonitor


def _running(pid):
    try:
        return psutil.Process(pid).is_running()
    except psutil.NoSuchProcess:
        return False


class TestProcessMonitorTreeKill(unittest.TestCase):

    def test_close_kills_descendants(self):
        pid_file = tempfile.NamedTemporaryFile(delete=False)
        pid_file.close()
        monitor = ProcessMonitor()
        parent_pid = None
        child_pid = None
        try:
            code = (
                'import subprocess, sys, time\n'
                'child = subprocess.Popen([sys.executable, "-c", '
                '"import time; time.sleep(120)"])\n'
                'open(sys.argv[1], "w", encoding="utf-8").write(str(child.pid))\n'
                'time.sleep(120)\n'
            )
            monitor.popen([sys.executable, '-c', code, pid_file.name])
            parent_pid = monitor._process.pid

            deadline = time.time() + 10
            while time.time() < deadline and not os.path.getsize(pid_file.name):
                time.sleep(0.05)
            self.assertTrue(os.path.getsize(pid_file.name), 'child pid was not written')

            with open(pid_file.name, encoding='utf-8') as handle:
                child_pid = int(handle.read().strip())

            self.assertTrue(_running(parent_pid))
            self.assertTrue(_running(child_pid))

            monitor.close()

            deadline = time.time() + 5
            while time.time() < deadline and (_running(parent_pid) or _running(child_pid)):
                time.sleep(0.05)

            self.assertFalse(_running(parent_pid))
            self.assertFalse(_running(child_pid))
        finally:
            for pid in (child_pid, parent_pid):
                if not pid or not _running(pid):
                    continue
                try:
                    psutil.Process(pid).kill()
                except psutil.Error:
                    pass
            try:
                os.remove(pid_file.name)
            except OSError:
                pass
