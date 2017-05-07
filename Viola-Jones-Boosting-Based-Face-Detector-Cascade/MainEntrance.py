import numpy as np
from scipy import misc
from time import time
import os
import pickle
import util
import AdaFeatureSelection as AFS
from pathos.multiprocessing import ProcessingPool as Pool


def GetFilter(wdw_stride,port):
    d = 64
    filters = []
    filt_sum = 0
    for w in range(2, d+1, wdw_stride):
        if w % 2 == 0:
            for h in range(1, d+1, wdw_stride):
                for i in range(1, d+1-h+1, max(1,int(h*port))):
                    for j in range(1, d+1-w+1, max(1,int(w*port))):
                        filters.append([i, j, h, w, 1])
                        filt_sum += 1

    for h in range(2, d+1, wdw_stride):
        if h % 2 == 0:
            for w in range(1, d+1, wdw_stride):
                for i in range(1, d+1-h+1, max(1,int(h * port))):
                    for j in range(1, d+1-w+1, max(1,int(w * port))):
                        filters.append([i, j, h, w, 2])
                        filt_sum += 1
    print(filt_sum)
    return(filters)



if __name__ == '__main__':
    d = 64
    n = 2000

    # read training images and convert to grayscale
    train_integral = np.zeros((2 * n, d + 1, d + 1))
    train_label = np.zeros((2 * n,))
    i = 0
    ir_face = util.ImageReader("./data/faces")
    for image in ir_face:
        train_integral[i, 1:, 1:] = util.GetIntegralImage(image)
        train_label[i] = 1
        i += 1
    ir_bg = util.ImageReader("./data/background")
    for image in ir_bg:
        train_integral[i, 1:, 1:] = util.GetIntegralImage(image)
        train_label[i] = -1
        i += 1

    # if we have trained classifiers, upload them from file
    if os.path.exists("./boosters.pkl"):
        print("boosters exist")
        bstrs = open("./boosters.pkl", 'rb')
        boosters = []
        num_booster = 4
        for i in range(num_booster):
            booster = pickle.load(bstrs)
            boosters.append(booster)

        # predict on training set
        pred_fc = (np.ones((2 * n,)) == 1)
        for i in range(num_booster):
            pred = boosters[i].predict(train_integral[pred_fc, :, :])
            y = train_label[pred_fc]
            wrong = (pred != y)
            train_error = np.sum(wrong) / len(y)
            print("traning crror: %.5f" % (train_error))
            pred_fc[pred_fc == True] = (pred == 1)

        print("False Positive: %d" % (sum(pred_fc & (train_label == -1))))
        print("True Positive: %d" % (sum(pred_fc & (train_label == 1))))
        print("True Negative: %d" % (sum((pred_fc == False) & (train_label == -1))))
        print("False Negative: %d" % (sum((pred_fc == False) & (train_label == 1))))

    # if we do not have trained classfiers, train
    else:
        print("boosters do not exist")

        print("start finding features")

        # contruct all filters
        wdw_stride = 2
        port = 0.75
        filters = GetFilter(wdw_stride, port)

        # parallel computing setup
        pool = Pool()
        pool.ncpus = 16

        # train classifiers and dump to file
        bstrs = open("./boosters.pkl", 'wb')
        tr_time = open("./train_time.pkl", 'wb')
        boosters = []
        num_booster = 4
        pred_fc = (np.ones((2 * n,)) == 1)
        for i in range(num_booster):
            boosters.append(AFS.AdaFeatureSelection(None, None))
            stime = time()
            boosters[-1].train(train_integral[pred_fc, :, :], train_label[pred_fc], filters, pool)
            pred = boosters[-1].predict(train_integral[pred_fc, :, :])
            pred_fc[pred_fc == True] = (pred == 1)
            duration = (time() - stime) / 60
            print("classifier %i: training time %.3f min" % (i+1, duration))
            pickle.dump(duration, tr_time)
            pickle.dump(boosters[-1], bstrs)

        bstrs.close()
        tr_time.close()

        # predict on training set
        pred_fc = (np.ones((2 * n,)) == 1)
        for i in range(num_booster):
            pred = boosters[i].predict(train_integral[pred_fc, :, :])
            pred_fc[pred_fc == True] = (pred == 1)

        print("False Positive: %d" % (sum(pred_fc & (train_label == -1))))
        print("True Positive: %d" % (sum(pred_fc & (train_label == 1))))
        print("True Negative: %d" % (sum((pred_fc == False) & (train_label == -1))))
        print("False Negative: %d" % (sum((pred_fc == False) & (train_label == 1))))

    # read the test image
    ir_test = util.ImageReader("./data/test")
    i = 0
    for image in ir_test:
        if i == 0:
            test_im = image
    h, w = test_im.shape

    IsFace = (np.ones((h - d + 1, w - d + 1)) == 1)

    # parse the test image and make prediction
    stime = time()
    ip = util.ImageParser([d, d], 1, test_im)
    k = 0
    for test_integral in ip:
        pred_fc = (np.ones((test_integral.shape[0],)) == 1)
        for i in range(num_booster):
            pred = boosters[i].predict(test_integral[pred_fc, :, :])
            pred_fc[pred_fc == True] = (pred == 1)
        IsFace[k, :] = pred_fc
        k += 1
    duration = (time() - stime) / 60
    print("duration = [%.3f mins]" % (duration))

    # label and draw faces
    fl = util.FaceLabeler(1280, 1600)
    labeled = fl.label(test_im, IsFace, d)
    misc.imsave("./labeled.jpg", labeled)