import model
import numpy as np

def test_train():
    myModel = model.ModelAPI()
    myModel.update_data()

    print "model initialized"

    mb_size = 256

    for i in range(0, 100000): 
        indices = np.random.choice(range(50000), mb_size, replace = False)
        scaling_factors = np.ones(mb_size)


        myModel.master_process_minibatch(indices, scaling_factors, "train")

        if i % 1000 == 0:
            print myModel.worker_process_minibatch(np.asarray(range(1000)), "train", ["loss"])

def all_tests():
    test_train()

