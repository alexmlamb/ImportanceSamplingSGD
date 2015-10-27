import model
import numpy as np
from config import get_config

def test_train():

    config = get_config()

    myModel = model.ModelAPI(config)


    print "model initialized"

    mb_size = 128

    for i in range(0, 100000): 
        indices = np.random.choice(range(574000), mb_size, replace = False)
        scaling_factors = np.ones(mb_size)


        myModel.master_process_minibatch(indices, scaling_factors, "train")

        if i == 2000:
            print "SAVING VALUES"
            paramStr = myModel.get_serialized_parameters()

        if i == 8000:
            print "LOADING OLD VALUES"
            myModel.set_serialized_parameters(paramStr)

        if i % 1000 == 0:
            loss = 0.0
            acc = 0.0
            for j in range(26):
                output = myModel.worker_process_minibatch(np.asarray(range(j * 1000, (j + 1) * 1000)), "test", ["loss", "accuracy"])
                loss += output["loss"].mean()
                acc += output["accuracy"]

            print "loss", loss / 26.0
            print "acc", acc / 26.0

def all_tests():
    test_train()

