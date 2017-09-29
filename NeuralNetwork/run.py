import numpy as np
import framework as fw
from time import time
import pickle
import os

if __name__ == '__main__':
    if not os.path.exists("./models"):
        os.mkdir("./models")

    ### read features and labels
    trainX = np.genfromtxt("./data/TrainDigitX.csv", delimiter=",")  # (50000, 784)
    trainY_raw = np.genfromtxt("./data/TrainDigitY.csv", delimiter=",", dtype=np.int32)  # (50000, )

    TestX = np.genfromtxt("./data/TestDigitX.csv", delimiter=",")  # (10000, 784)
    TestX2 = np.genfromtxt("./data/TestDigitX2.csv", delimiter=",")  # (5000, 784)
    TestY_raw = np.genfromtxt("./data/TestDigitY.csv", delimiter=",", dtype=np.int32)  # (10000, )

    feature_dim = trainX.shape[1]
    num_class = 10

    ### convert labels to one-hot vectors
    num_Train = trainY_raw.shape[0]
    trainY = np.zeros((num_Train, num_class))
    index = np.array(range(num_Train))
    trainY[index, trainY_raw] = 1

    num_Test = TestY_raw.shape[0]
    TestY = np.zeros((num_Test, num_class))
    index = np.array(range(num_Test))
    TestY[index, TestY_raw] = 1

    holdout_rate = 0.2
    num_Holdout = int(num_Train * holdout_rate)
    num_Train -= num_Holdout

    ### separate holdout set
    np.random.seed(123)
    index = np.random.choice(np.array(range(num_Train + num_Holdout)), size=num_Holdout, replace=False)
    mask = np.ones((num_Train + num_Holdout,), dtype=bool)
    mask[index] = False
    TrainX = trainX[mask,]
    TrainY = trainY[mask,]
    HoldoutX = trainX[index,]
    HoldoutY = trainY[index,]

    batch_size = 128
    hidden_dims = [32, 64, 128, 256]
    etas = [0.05, 0.10, 0.30, 0.50]
    for hidden_dim in hidden_dims:
        for eta in etas:
            print("")
            print("")
            print("")
            print("hidden_dim: %d; eta: %.4f" % (hidden_dim, eta))
            print("")

            ### build neuro net computation graph
            fw.parameters = []
            fw.nodes = []

            paras = []
            Input = fw.Value(None, (1, feature_dim))
            Label = fw.Value(None, (1, num_class))

            W1 = fw.Para((feature_dim, hidden_dim))
            b1 = fw.Para((hidden_dim,))
            hidden1 = fw.Sigmoid(fw.VAdd(fw.VMul(Input, W1), b1))

            W2 = fw.Para((hidden_dim, hidden_dim))
            b2 = fw.Para((hidden_dim,))
            hidden2 = fw.Sigmoid(fw.VAdd(fw.VMul(hidden1, W2), b2))

            W3 = fw.Para((hidden_dim, num_class))
            b3 = fw.Para((num_class,))
            pred = fw.Softmax(fw.VAdd(fw.VMul(hidden2, W3), b3))

            loss = fw.Mean(fw.SquaredLoss(pred, Label))
            paras.extend([W1, b1, W2, b2, W3, b3])

            ### train
            epoch = 1
            train_loss = []
            train_error = []
            holdout_loss = []
            holdout_error = []
            holdout_error_min = fw.DataType(1)
            while True:
                stime = time()
                perm = np.random.permutation(num_Train)
                num_fed = 0
                while num_fed < num_Train:
                    feed = min(batch_size, num_Train - num_fed)
                    Input.set(TrainX[perm[num_fed: num_fed + feed], ], (feed, feature_dim))
                    Label.set(TrainY[perm[num_fed: num_fed + feed], ], (feed, num_class))
                    num_fed += feed
                    fw.Forward()
                    fw.Backward(loss)
                    fw.SGD(eta)
                # predict on training set
                Input.set(TrainX, (num_Train, feature_dim))
                Label.set(TrainY, (num_Train, num_class))
                fw.Forward()
                train_loss.append(loss.value)
                train_error.append(1. - fw.accuracy(pred.value, Label.value))
                # predict on holdout set
                Input.set(HoldoutX, (num_Holdout, feature_dim))
                Label.set(HoldoutY, (num_Holdout, num_class))
                fw.Forward()
                holdout_loss.append(loss.value)
                holdout_error.append(1. - fw.accuracy(pred.value, Label.value))
                duration = time() - stime
                print("Epoch %d: train loss = %.4f; holdout loss = %.4f; holdout error = %.4f  [%.3f secs]" % (
                epoch, train_loss[epoch - 1], holdout_loss[-1], holdout_error[-1], duration))
                if holdout_error[-1] - holdout_error_min > 0.05 * holdout_error_min:
                    break
                if epoch >= 100:
                    break
                else:
                    if holdout_error[-1] < holdout_error_min:
                        f = open('./models/model_dim' + str(hidden_dim) + 'eta' + str(eta) + '.pkl', 'wb')
                        p_value = []
                        for p in paras:
                            p_value.append(p.value)
                        pickle.dump(p_value, f)
                        f.close()
                    holdout_error_min = np.minimum(holdout_error[-1], holdout_error_min)
                    epoch += 1

            with open('./models/model_dim' + str(hidden_dim) + 'eta' + str(eta) + '.pkl', 'rb') as f:
                p_value = pickle.load(f)
                idx = 0
                for p in p_value:
                    paras[idx].value = p
                    idx += 1

            ### predict on test set
            Input.set(TestX, (num_Test, feature_dim))
            Label.set(TestY, (num_Test, num_class))
            fw.Forward()
            test_loss = loss.value
            test_error = 1. - fw.accuracy(pred.value, Label.value)
            print("Test loss = %.4f; Test error = %.4f" % (test_loss, test_error))
