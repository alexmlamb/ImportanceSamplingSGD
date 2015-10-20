
import redis
import subprocess
import numpy as np

import re
import os

# maybe add option to resume from something ?


class EphemeralRedisServer():

    def __init__(self, scratch_path, port=None, password=None):
        # The use of a password would be to prevent accidents
        # from experiments talking to each other.

        self.scratch_path = scratch_path
        self.port = port
        self.password = password

        assert os.path.exists(self.scratch_path)
        self.dbfilename = "%0.3d_%0.5d.rdb" % (np.random.randint(low=0, high=1000), self.port)

        self.server_process = None

        EphemeralRedisServer.assert_redis_server_executable_is_found()

    @staticmethod
    def assert_redis_server_executable_is_found():

        cmd = "which redis-server"
        #cmd = "ls"
        try:
            res = subprocess.check_output(cmd, shell=True)
            L_res = [e for e in res.split('\n') if 0 < len(e)]
            error_msg = "Could not find redis-server. `which redis-server` yields : %s" % res
            assert len(L_res) == 1, error_msg
            assert re.match(".*/redis-server", L_res[0]), error_msg
        except:
            print "We couldn't even call the following command successfully."
            print cmd
            quit()

    def start(self):

        if self.password is not None:
            password_cmd_str = "  --requirepass '%s'" % self.password
        else:
            password_cmd_str = "  "

        port_cmd_str = "  --port %d" % self.port

        dir_cmd_str = "  --dir '%s'" % self.scratch_path
        dbfilename_cmd_str = "  --dbfilename '%s'" % self.dbfilename

        # we don't care about our data in the event of a crash
        appendfsync_cmd_str = "  --appendfsync no"
        maxclients_cmd_str = "  --maxclients 8"


        L_cmd_str = ["redis-server", password_cmd_str, port_cmd_str, dir_cmd_str, dbfilename_cmd_str, appendfsync_cmd_str, maxclients_cmd_str]
        cmd = "".join(L_cmd_str)

        print "\n".join(L_cmd_str)
        #print "Running subprocess: \n\t%s" % cmd

        # The os.setsid() is passed in the argument preexec_fn so
        # it's run after the fork() and before  exec() to run the shell.
        self.server_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                                shell=True, preexec_fn=os.setsid)

        print "redis server now running with pid %d" % self.server_process.pid

    def stop(self):

        if self.server_process is None:
            print "The server is not running. Calling `stop` does nothing."

        if self.password is not None:
            password_cmd_str = "  -a '%s'" % self.password
        else:
            password_cmd_str = "  "

        port_cmd_str = "  -p %d" % self.port

        cmd = "redis-cli %s %s shutdown" % (password_cmd_str, port_cmd_str)
        print cmd
        res = subprocess.check_output(cmd, shell=True)
        print res


        res = self.server_process.communicate(input=None)
        self.server_process.wait()
        print "Waited for process pid %d." % self.server_process.pid
        print "STDOUT was :"
        for line in res[0].split("\n"):
            print line
        print "STDERR was :"
        for line in res[1].split("\n"):
            print line

    def get_client(self):
        return redis.StrictRedis(host='localhost', port=self.port, password=self.password)

