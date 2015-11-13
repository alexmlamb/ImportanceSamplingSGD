
import time
import pickle
import json
import numpy as np

from collections import defaultdict

class Logger(object):

    def __init__(self):

        self.DL_logs = defaultdict(list)
        self.closed = False

    def log(self, channel, e):
        # Assumes that `e` is something that can be converted into json.
        # These calls are expected to occur for a lot of channels that
        # are not doing to be monitored.

        if self.closed:
            return

        self.DL_logs[channel].append((time.time(), e))

    def close(self):
        self.closed = True

    def save_to_pickle(self, path):
        pickle.dump(self.DL_logs, open(path, "w"))


class RedisLogger(Logger):

    def __init__(self, rsconn, queue_prefix_identifier=None):

        super(RedisLogger, self).__init__()

        # `rsconn` is a connection to a redis database. It must be currently
        # active/connected and should not be closed before this logger gets a chance
        # to call `commit_and_clear`, or otherwise you will lose the latest events
        # logged.
        #
        # `queue_prefix_identifier` is a name given to the queue on the redis server
        # to which we will append the values. It really should be unique, because this
        # will be used to identify the entity committing things to the logs.

        self.rsconn = rsconn

        if queue_prefix_identifier in ['service_database', 'service_master', 'service_worker']:
            self.queue_prefix_identifier = "logging/" + queue_prefix_identifier + "/" + str(np.random.randint(low=0, high=np.iinfo(np.uint32).max))

        if queue_prefix_identifier is None:
            self.queue_prefix_identifier = "logging/" + str(np.random.randint(low=0, high=np.iinfo(np.uint32).max))

        self.database_name_for_L_queue_prefix_identifier = "logging:L_queue_prefix_identifier"
        self.database_name_for_L_channels = "logging:S_channels"
        self.rsconn.rpush(self.database_name_for_L_queue_prefix_identifier, self.queue_prefix_identifier)

        #for channel in self.L_channels:
        #    if channel not in self.L_channels_currently_logged_on_database:
        #        self.rsconn.rpush(self.database_name_for_L_channels, "%s/%s" % (self.queue_prefix_identifier, channel)

        # sync to database automatically after 30 seconds
        self.auto_sync_period = 30
        self.last_sync_timestamp = None

    def log(self, channel, e):
        # The point of having `forbid_sync` is that we don't want to
        # have infinite recursion due to the fact that we log the event
        # of database synchronization (which would otherwise potentially
        # trigger another synchronization if the user had
        # self.auto_sync_period = 0, which is a bad idea).
        super(RedisLogger, self).log(channel, e)
        if self.closed:
            return
        else:
            self._auto_sync_if_necessary()

    def close(self):
        super(RedisLogger, self).close()
        self.commit_and_clear()

    def commit_and_clear(self):
        for channel in self.DL_logs.keys():
            queue_name_for_that_channel = "%s/%s" % (self.queue_prefix_identifier, channel)
            for e in self.DL_logs[channel]:
                self.rsconn.rpush(queue_name_for_that_channel, json.dumps(e))
            # clear it out now
            del self.DL_logs[channel]
            # make sure it's recorded in the database so we know how to get it
            self.rsconn.sadd(self.database_name_for_L_channels, channel)

    def _auto_sync_if_necessary(self):
        tic = time.time()
        if (self.last_sync_timestamp is None or
            self.last_sync_timestamp + self.auto_sync_period <= tic):
            tic = time.time()
            self.commit_and_clear()
            toc = time.time()
            self.last_sync_timestamp = toc
            self.log('timing_profiler', {'commit_and_clear' : (toc-tic)})



def record_machine_info(remote_redis_logger):

    D_info = {}

    # This should be a method that's shared with all the 3 components.
    import socket
    D_info['hostname'] = socket.gethostname()

    # We don't want to load theano unless this process is going to use theano.
    # That would just be a bad idea.
    import sys
    sys.modules.keys()

    if 'theano' in sys.modules:
        import theano
        D_info['theano.config.device'] = theano.config.device
        D_info['theano.config.floatX'] = theano.config.floatX

    remote_redis_logger.log('machine_info', D_info)
