__author__ = 'chinna'

import numpy as np
#import matplotlib.pyplot as plt
import re
import math
import matplotlib

def conv_number(s):
    try:
        s = float(s)
        if math.isnan(s):
            return 0
        return s
    except ValueError:
        return 0

def get_config():
    data_root = '/u/lambalex/DeepLearning/ImportanceSampling/logs/'
    #data_root = '/Users/chinna/ImportanceSamplingSGD_alex/logging/'
    config = {}
    config['files'] = {'usgd': data_root + 'log_1446969742_.txt',
                       'isgd': data_root + 'log_1447143588_.txt',
                       'isgd_bug' : data_root + 'log_1447060083_.txt'}
    config['plot'] = '/u/sankarch/Documents/ImportanceSamplingSGD/model_protoype/exp_chinna/' + 'plot.jpeg'
    return config

class parser_class:

    def __init__(self):
        print "creating parser object"
        self.L_accuracy = {}
        self.config = get_config()

    def update_L_accuracy(self,line,L_accuracy):
        match = re.search(r'\s*(accuracy)\s*:\s*(mean)\s*(?P<accuracy>\S*)\s*,',line)
        if match:
            acc = match.group('accuracy')
            L_accuracy.append(conv_number(acc))

    def update_L_mini_batch(self,line,L_mini_batch):
        match = re.search(r'Number minibatches processed by master\s*(?P<mini_batch_num>\S*)',line)
        if match:
            acc = match.group('mini_batch_num')
            L_mini_batch.append(conv_number(acc))

    def parse_file(self,file):
        fd = open(file,'r')
        L_accuracy = []
        for line in fd:
            self.update_L_accuracy(line,L_accuracy)
            #self.update_L_mini_batch(line)
        print "finished parsing",file,len(L_accuracy)
        return L_accuracy

    def parse(self):
        for f_type,file in self.config['files'].items():
            self.L_accuracy[f_type] = self.parse_file(file)

    def plot(self):
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        #match = re.match(r'(log_)(\S).txt',self.file)
        #plt_file = match.group(2)
        for k,v in self.L_accuracy.items():
            plt.plot(v, label=k)
        plt.legend()
        plt.savefig(self.config['plot'])
        print "plot saved to",self.config['plot']

    def run(self):
        self.parse()
        self.plot()

if __name__ == "__main__":
    parser = parser_class()
    parser.run()







"""
        fd = self.readfile(file)
        s = "---- accuracy : mean 983.23, std nan    with 0.0000 of values used"
        match = re.search(r'\s*(accuracy)\s*:\s*(mean)\s*(?P<accuracy>\S*)\s*,',s)
        print float(match.group('accuracy')) + 1.0
        s = 'Number minibatches processed by master    92342'
        match = re.search(r'Number minibatches processed by master\s*(?P<mini_batch_num>\S*)',s)
        print match.group('mini_batch_num')"""
