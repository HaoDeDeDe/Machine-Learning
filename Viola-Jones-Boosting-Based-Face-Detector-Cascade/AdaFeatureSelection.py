import numpy as np
from time import time
import RddDecisionStump as RDS
import DecisionStump as DS
from pathos.multiprocessing import ProcessingPool as Pool

class AdaFeatureSelection:
    def __init__(self, baseClf, Theta):
        self.baseClf = baseClf
        self.Theta = Theta

    def InitDistri(self, y):
        n = y.shape[0]
        pos = (y == 1)
        neg = (y == -1)
        D = np.ones((n,))
        D[pos] = float(1) / (2 * sum(pos))
        D[neg] = float(1) / (2 * sum(neg))
        return (D)

    def GetWeightAndDistr(self, error, yhat, y, D):
        alpha = np.log((1 - error) / error) / 2
        Z = 2 * np.sqrt(error * (1 - error))
        D = D * np.exp(-1 * alpha * y * yhat) / Z
        return ([alpha, D])

    def GetOneFeature(self, x, y, D, filters, pool):
        n, d, d = x.shape

        # experiment decision stump
        exper_DS = RDS.RddDecisionStump(x, y, D)

        results = pool.map(exper_DS.train, filters)
        best = min(results, key=lambda a: a[0])

        return(best)


    def train(self, x, y, filters, pool):
        n = x.shape[0]

        D = self.InitDistri(y)
        self.baseClf = []
        T = 0

        discri = np.zeros((n,))
        while True:
            stime = time()
            error, theta, p, filt = self.GetOneFeature(x, y, D, filters, pool)
            self.baseClf.append(DS.DecisionStump(theta, p, filt, None))
            yhat = self.baseClf[-1].predict(x)
            alpha, D = self.GetWeightAndDistr(error, yhat, y, D)
            self.baseClf[-1].SetAlpha(alpha)

            T += 1
            duration = (time() - stime) / 60

            discri += self.baseClf[-1].alpha * yhat
            args = discri.argsort()
            ytmp = y[args]
            yhat = discri[args]
            Theta = (yhat[ytmp == 1])[0]
            FPR = sum((yhat >= Theta) & (ytmp == -1)) / float(sum(ytmp == -1))  ## FPR = FP / (FP+TN) = FP / N
            print("T: %i;  error:%f;   alpha: %.5f;  p: %i;  theta:%.3f;  Theta: %.3f;  FPR: %.5f;  duration = [%.3f mins]" % \
                (T, error, alpha, self.baseClf[-1].p, self.baseClf[-1].theta, Theta, FPR, duration))
            if FPR <= 0.3:
                self.Theta = Theta
                break

    def predict(self, x):
        n = x.shape[0]
        T = len(self.baseClf)
        yhat = np.zeros((n,))
        for t in range(T):
            a = self.baseClf[t].predict(x)
            yhat += self.baseClf[t].alpha * self.baseClf[t].predict(x)
        yhat = np.sign(yhat - self.Theta)
        yhat[yhat == 0] = 1
        return (yhat)
