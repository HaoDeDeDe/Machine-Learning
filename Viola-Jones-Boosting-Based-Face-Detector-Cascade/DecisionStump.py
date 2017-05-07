import numpy as np


class DecisionStump:
    def __init__(self, theta, p, filt, alpha):
        self.theta = theta
        self.p = p
        self.alpha = alpha
        self.filt = filt

    def SetAlpha(self, alpha):
        self.alpha = alpha

    def SetFilter(self, filt):
        self.filt = filt

    def GetFeature(self, x):
        n = x.shape[0]

        i = int(self.filt[0])
        j = int(self.filt[1])
        h = int(self.filt[2])
        w = int(self.filt[3])
        w2 = int(w / 2)
        h2 = int(h / 2)
        Type = int(self.filt[4])

        num = w * h / 2
        if Type == 1:
            pos = x[:, i + h - 1, j + w2 - 1] + x[:, i - 1, j - 1] - x[:, i - 1, j + w2 - 1] - x[:, i + h - 1, j - 1]
            neg = x[:, i + h - 1, j + w - 1] + x[:, i - 1, j + w2 - 1] - x[:, i - 1, j + w - 1] - x[:, i + h - 1,
                                                                                                  j + w2 - 1]
            feature = (pos / num - neg / num).reshape((n,))
        else:
            pos = x[:, i + h2 - 1, j + w - 1] + x[:, i - 1, j - 1] - x[:, i - 1, j + w - 1] - x[:, i + h2 - 1, j - 1]
            neg = x[:, i + h - 1, j + w - 1] + x[:, i + h2 - 1, j - 1] - x[:, i + h2 - 1, j + w - 1] - x[:, i + h - 1,
                                                                                                       j - 1]
            feature = (pos / num - neg / num).reshape((n,))
        return (feature)

    def predict(self, x):
        x = self.GetFeature(x)
        yhat = np.sign(self.p * (x - self.theta))
        yhat[yhat == 0] = 1
        return (yhat)