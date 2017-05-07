import numpy as np
import scipy
import matplotlib.pyplot as plt
import random

# normalize each example to unit norm
def normalizeExample(data):
    n, d = data.shape
    normalized = data / np.sqrt(np.sum(data, axis=1)).reshape((n,1))
    return(normalized)

# one step in perceptron online training
def PerceptronOnlineTrain(feature, label, w):
    score = np.matmul(feature, w)
    pred = int(score>=0) * 2 - 1
    mistake = 0
    if pred == -1 and label == 1:
        w += np.transpose(feature).reshape(w.shape)
        mistake = 1
    if pred == 1 and label == -1:
        w -= np.transpose(feature).reshape(w.shape)
        mistake = 1
    return([w, mistake])

# one step in perceptron batch training
def PerceptronBatchTrain(features, labels, M):
    n, d = features.shape
    w = np.zeros((d,1))
    mistakes = [0]
    for m in range(M):
        for i in range(n):
            w, mistake = PerceptronOnlineTrain(features[i,:], labels[i], w)
            mistakes.append(mistakes[-1]+mistake)
    mistakes = np.array(mistakes)
    return([w, mistakes])

# make prediction
def PerceptronPredict(features, w):
    n, d = features.shape
    scores = np.matmul( features, w)
    pred = -1 * np.ones((n,1))
    pred[scores>=0] = 1
    return(pred)

# calculate prediction error
def PredictionError(yhat, y):
    n = len(y)
    wrong = np.ones((n,1))
    error = sum(wrong[yhat[:,0]!=y]) / len(y)
    return(error)

# cross validation to get optimal M
def PerceptronCV(features, labels, K, lower, upper):
    n = len(labels)
    Kfold = np.random.choice(range(n), n, replace=False)
    foldSize = int(n/K)
    Ms = range(lower,upper+1)
    CVerrors = np.zeros(len(Ms))
    j = 0
    for M in Ms:
        for i in range(K):
            trainFeatures = np.concatenate((features[Kfold[0:i*foldSize],:], features[Kfold[(i+1)*foldSize:n],:]), axis=0)
            trainLabels = np.concatenate((labels[Kfold[0:i * foldSize]], labels[Kfold[(i + 1) * foldSize:n]]), axis=0)
            if i==K-1:
                validFeatures = features[Kfold[i*foldSize:n],:]
                validLabels = labels[Kfold[i*foldSize:n]]
            else:
                validFeatures = features[Kfold[i * foldSize: (i+1)*foldSize], :]
                validLabels = labels[Kfold[i * foldSize: (i + 1) * foldSize]]
            [w, mistakes] = PerceptronBatchTrain(trainFeatures, trainLabels, M)
            pred = PerceptronPredict(validFeatures, w)
            CVerrors[j] += PredictionError(pred, validLabels)
        CVerrors[j] /= K
        j += 1
    return([Ms, CVerrors])



if __name__ == '__main__':
    train_digits = np.loadtxt("./train35.digits")
    train_labels = np.loadtxt("./train35.labels")
    test_digits = np.loadtxt("./test35.digits")

    # normalize each example
    train_digits = normalizeExample(train_digits)
    test_digits = normalizeExample(train_digits)

    random.seed(100)

    # do cross validation to get optimal M
    [Ms, CVerrors] = PerceptronCV(train_digits, train_labels, 5, 1, 10)
    plt.scatter(Ms, CVerrors, color="green")
    plt.title("CV errors versus M")
    plt.xlabel("M")
    plt.ylabel("CV error")
    plt.show()

    # with optimal M, retrain the perceptron using the whole training set
    M = Ms[np.argmin(CVerrors)]
    [w, mistakes] = PerceptronBatchTrain(train_digits, train_labels, M)
    train_pred = PerceptronPredict(train_digits, w)
    error = PredictionError(train_pred, train_labels)
    plt.plot(np.array(range(0,len(mistakes))), mistakes)
    plt.xlabel("number of examples seen")
    plt.ylabel("number of cumulative mistakes")
    plt.title("frequency of mistakes")
    plt.show()

    # make prediciton on test set
    test_pred = PerceptronPredict(test_digits, w)
    np.savetxt("./test35.predictions", test_pred, fmt='%i')



