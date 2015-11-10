__author__ = 'chinna'
import theano
from theano import tensor as T
import numpy as np
import matplotlib.pyplot as plt
from theano import shared
from theano import function
import scipy as sp
from scipy import signal
from PIL import Image
import re
import math

def conv_number(s):
    try:
        s = float(s)
        if math.isnan(s):
            return 0
        return s
    except ValueError:
        return 0

class parser_class:

    def __init__(self,file):
        print "creating parser object"
        self.L_accuracy = []
        self.L_mini_batch =[]
        self.file = file

    def update_L_accuracy(self,line):
        match = re.search(r'\s*(accuracy)\s*:\s*(mean)\s*(?P<accuracy>\S*)\s*,',line)
        if match:
            acc = match.group('accuracy')
            self.L_accuracy.append(conv_number(acc))

    def update_L_mini_batch(self,line):
        match = re.search(r'Number minibatches processed by master\s*(?P<mini_batch_num>\S*)',line)
        if match:
            acc = match.group('mini_batch_num')
            self.L_mini_batch.append(conv_number(acc))

    def parse(self):
        fd = open(self.file,'r')
        for line in fd:
            self.update_L_accuracy(line)
            self.update_L_mini_batch(line)
        print "finished parsing"

    def plot(self):
        #match = re.match(r'(log_)(\S).txt',self.file)
        #plt_file = match.group(2)
        plt_file = '1'
        plt_file = 'plot'+ plt_file + '.jpeg'
        plt.plot(self.L_accuracy[0::3])
        plt.plot(self.L_accuracy[1::3])
        plt.plot(self.L_accuracy[2::3])
        plt.savefig(plt_file)
        print "plot saved to",plt_file

    def run(self):
        self.parse()
        self.plot()

if __name__ == "__main__":
    file = "/Users/chinna/ImportanceSamplingSGD_alex/logging/log_1447166866_.txt"
    parser = parser_class(file)
    parser.run()







"""
        fd = self.readfile(file)
        s = "---- accuracy : mean 983.23, std nan    with 0.0000 of values used"
        match = re.search(r'\s*(accuracy)\s*:\s*(mean)\s*(?P<accuracy>\S*)\s*,',s)
        print float(match.group('accuracy')) + 1.0
        s = 'Number minibatches processed by master    92342'
        match = re.search(r'Number minibatches processed by master\s*(?P<mini_batch_num>\S*)',s)
        print match.group('mini_batch_num')"""