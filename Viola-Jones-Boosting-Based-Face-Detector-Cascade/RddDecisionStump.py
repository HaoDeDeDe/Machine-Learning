import numpy as np


class RddDecisionStump:
    def __init__(self, inp, y, D):
        self.inp = inp
        self.y = y
        self.D = D

    def SetInput(self, inp):
        self.inp = inp

    def SetY(self, y):
        self.y = y

    def SetDataDistri(self, D):
        self.D = D

    def GetFeature(self, filt):
        n = self.inp.shape[0]

        i = int(filt[0])
        j = int(filt[1])
        h = int(filt[2])
        w = int(filt[3])
        w2 = int(w / 2)
        h2 = int(h / 2)
        Type = int(filt[4])

        num = w * h / 2
        if Type == 1:
            pos = self.inp[:,i+h-1, j+w2-1] + self.inp[:, i-1, j-1] - self.inp[:,i-1,j+w2-1] - self.inp[:,i+h-1,j-1]
            neg = self.inp[:,i+h-1, j+w-1] + self.inp[:, i-1, j+w2-1] - self.inp[:,i-1, j+w-1] - self.inp[:,i+h-1,j+w2-1]
            feature = (pos / num - neg / num).reshape((n,))
        else:
            pos = self.inp[:, i+h2-1, j+w-1] + self.inp[:, i-1, j-1] - self.inp[:, i-1,j+w-1] - self.inp[:,i+h2-1, j-1]
            neg = self.inp[:, i+h-1, j+w-1] + self.inp[:, i+h2-1, j-1] - self.inp[:, i+h2-1, j+w-1] - self.inp[:, i+h-1, j-1]
            feature = (pos / num - neg / num).reshape((n,))

        return (feature)

    def train(self, filt):
        x = self.GetFeature(filt)
        args = x.argsort()
        y = self.y[args]
        D = self.D[args]
        x = x[args]
        n = len(x)
        pos = (y == 1)
        neg = (y == -1)
        Tpos = sum(D[pos])
        Tneg = sum(D[neg])

        Spos = np.cumsum(pos*D)
        Sneg = np.cumsum(neg*D)
        eps = np.minimum(Spos + Tneg - Sneg, Sneg + Tpos - Spos)
        eps = eps[0:(n-1)]
        J = np.argmin(eps)
        theta = (x[J] + x[J + 1]) / 2
        p = ([1, -1])[np.equal(Sneg[J] + Tpos - Spos[J], eps[J])]

        return ([np.min(eps), theta, p, filt])
