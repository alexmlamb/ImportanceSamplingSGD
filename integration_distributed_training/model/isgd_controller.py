import os
import time

import numpy as np

from config import get_config
from platoon.channel import Controller


class LSTMController(Controller):
    """
    This multi-process controller implements patience-based early-stopping SGD
    """

    def __init__(self, control_port, max_mb, valid_freq):
        """
        Initialize the LSTMController
        Parameters
        ----------
        max_mb : int
            Max number of minibatches to train on.
        valid_freq : int
            Number of minibatches to train on between every monitoring step.
        """
        Controller.__init__(self, control_port)

        self.max_mb = max_mb

        self.valid_freq = valid_freq
        self.nb_mb = 0

        self.valid = False
        self.start_time = None

        config = get_config()

        #self.experiment_dir = "{}exp_{}".format(config['plot_output_directory'], time.strftime("%Y-%m-%d_%H-%M-%S"))
        #os.mkdir(self.experiment_dir)

    def handle_control(self, req, worker_id):
        print "Received '{}' from {}. ".format(req, worker_id),
        control_response = ""

        if req == 'next':
            if self.start_time is None:
                self.start_time = time.time()

            if self.valid:
                self.valid = False
                control_response = 'valid'
            else:
                control_response = 'train'


        elif 'train_done' in req:
            self.nb_mb += req['train_done']
            if np.mod(self.nb_mb, self.valid_freq) == 0:
                self.valid = True

        elif 'valid_done' in req:
            print "\nTotal Loss: {}".format(req['valid_done'])
            print "Training time {:.2f}s".format(time.time() - self.start_time)
            print "Number of minibatches:", self.nb_mb

            if np.isnan(req['valid_done']):
                control_response = "stop"
                self.worker_is_done(worker_id)

        if self.nb_mb >= self.max_mb:
            control_response = 'stop'
            self.worker_is_done(worker_id)
            print "\nTraining time {:.2f}s".format(time.time() - self.start_time)
            print "Number of minibatches:", self.nb_mb

        print "Answering: {}.".format(control_response)
        return control_response

if __name__ == '__main__':
    l = LSTMController(control_port=4222, max_mb=5000, valid_freq=20)
    print "Controller is ready"
    l.serve()

