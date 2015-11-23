

def get_config():

    config = {}

    #Importance sampling or vanilla sgd.  
    config["importance_algorithm"] = "isgd"
    #config["importance_algorithm"] = "sgd"

    #Momentum rate, where 0.0 corresponds to not using momentum
    config["momentum_rate"] = 0.9

    #The learning rate to use on the gradient averaged over a minibatch
    config["learning_rate"] = 0.1

    #config["dataset"] = "mnist"
    config["dataset"] = "svhn"

    if config["dataset"] == "mnist":
        config["num_input"] = 784
    elif config["dataset"] == "svhn":
        config["num_input"] = 3072

    config["mnist_file"] = "/data/lisatmp4/lambalex/mnist/mnist.pkl.gz"
    config["svhn_file_train"] = "/data/lisatmp4/lambalex/svhn/train_32x32.mat"
    config["svhn_file_extra"] = "/data/lisatmp4/lambalex/svhn/extra_32x32.mat"
    config["svhn_file_test"] = "/data/lisatmp4/lambalex/svhn/test_32x32.mat"

    config["hidden_sizes"] = [2048, 2048,2048,2048]

    config["seed"] = 9999494

    config["learning_rate"] = 0.01

    #Weights are initialized to N(0,1) * initial_weight_size
    config["initial_weight_size"] = 0.01

    #Hold this fraction of the instances in the validation dataset
    config["fraction_validation"] = 0.05
    

    return config


