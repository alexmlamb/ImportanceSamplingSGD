

import redis
import numpy as np
import time

import progressbar

import signal
import sys

from redis_server_wrapper import EphemeralRedisServer


def start_redis_server(server_port=None):

    server_scratch_path = "."
    if server_port is None:
        server_port = np.random.randint(low=1025, high=65535)

    #server_password = "".join(["%d" % np.random.randint(low=0, high=10) for _ in range(10)])
    server_password = None

    rserv = EphemeralRedisServer(   scratch_path=server_scratch_path,
                                    port=server_port, password=server_password)

    rserv.start()
    time.sleep(5)
    rsconn = rserv.get_client()
    print "pinging master server : %s" % (rsconn.ping(),)

    import socket
    hostname = socket.gethostname()

    D_server_desc = {'hostname' : hostname, 'port' : server_port, 'password' : server_password}
    return (rserv, rsconn, D_server_desc)


def test_cycle_queue(rsconn, N=20):

    queue_name = "L_queue"
    for n in range(N):
        #value = n
        value = (n * np.ones(100000, dtype=np.int8)).tostring()
        rsconn.rpush(queue_name, value)

    Nread = rsconn.llen(queue_name)
    print "(N, Nread) is (%d, %d)." % (N, Nread)

    for _ in range(1000):
        for n in range(N):
            e = rsconn.lpop(queue_name)
            rsconn.rpush(queue_name, e)

    L = []
    while 0 < rsconn.llen(queue_name):
        e = rsconn.lpop(queue_name)
        L.append(e)

    print [np.fromstring(e, dtype=np.int8)[0] for e in L]




def test_timestamp_hashmap(rsconn):

    def get_next_timestamp():
        get_next_timestamp.counter += 1.0
        return get_next_timestamp.counter
    get_next_timestamp.counter = 0.0

    #def get_next_timestamp():
    #    return time.time()



    N = 100
    hashmap_name = "H_timestamps"
    D_ref = {}
    for n in range(N):
        #value = n
        value = (n * np.ones(100000, dtype=np.int8)).tostring()
        timestamp_str = str(get_next_timestamp())
        rsconn.hset(hashmap_name, value, timestamp_str)
        D_ref[value] = timestamp_str

    Niter = 1000
    widgets = ['Parsing lines: ', progressbar.Percentage(),
                ' ', progressbar.Bar(marker=progressbar.RotatingMarker()),
                ' ', progressbar.ETA()]
    pbar = progressbar.ProgressBar(widgets=widgets, maxval=Niter-1).start()

    previous_timestamp = time.time()
    for niter in range(Niter):
        for (k, local_recorded_timestamp_str) in D_ref.items():
            current_timestamp = get_next_timestamp()
            database_recorded_timestamp_str = rsconn.hget(hashmap_name, k)
            database_recorded_timestamp = float(database_recorded_timestamp_str)
            local_recorded_timestamp = float(local_recorded_timestamp_str)

            assert local_recorded_timestamp <= current_timestamp
            assert database_recorded_timestamp <= current_timestamp

            current_timestamp_str = str(current_timestamp)

            D_ref[k] = current_timestamp_str
            rsconn.hset(hashmap_name, k, current_timestamp_str)

            pbar.update(niter)


def run():

    (rserv, rsconn, _) = start_redis_server()

    def signal_handler(signal, frame):
        rserv.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    #test_cycle_queue(rsconn)
    test_timestamp_hashmap(rsconn)




if __name__ == "__main__":
    run()
