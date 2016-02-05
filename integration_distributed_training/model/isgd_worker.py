import model
import numpy as np
from config import get_config
from platoon.channel import Worker
from platoon.param_sync import EASGD
import theano
import theano.tensor as T

config = get_config()

myModel = model.ModelAPI(config)

print "model initialized"

mb_size = 128

worker = Worker(control_port=4222)
device = theano.config.device

platoon_sync_rule = EASGD(0.3)
nb_minibatches_before_sync = 10  # 10 from EASGD paper

params = myModel.nnet.parameters

for param in params:
    print param.get_value().dtype

worker.init_shared_params(params, param_sync_rule=platoon_sync_rule)

step = worker.send_req('next')

print "training started"

for i in range(0, 100000):
    indices = np.random.choice(range(574000), mb_size, replace = False)
    scaling_factors = np.ones(mb_size)

    step = worker.send_req('next')

    if step == "train":
        for i in xrange(nb_minibatches_before_sync):
            indices = np.random.choice(range(574000), mb_size, replace = False)
            scaling_factors = np.ones(mb_size)
            myModel.master_process_minibatch(indices, scaling_factors, "train")

            step = worker.send_req(dict(train_done=nb_minibatches_before_sync))
            print "Syncing with global params"
            worker.sync_params(synchronous=True)

    if step == "valid":

        worker.copy_to_local()

        loss = 0.0
        acc = 0.0
        numMB = 300
        for j in range(numMB):
            output = myModel.worker_process_minibatch(np.asarray(range(j * 1000, (j + 1) * 1000)), "train", ["loss", "accuracy"])
            loss += output["loss"].mean()
            acc += output["accuracy"]

        print "test loss", loss * 1.0 / numMB
        print "test acc", sum(acc) / (numMB * len(acc))







