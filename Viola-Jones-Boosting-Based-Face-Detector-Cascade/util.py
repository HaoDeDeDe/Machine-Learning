import numpy as np
from scipy import misc
import os
from sklearn import cluster
from sklearn import metrics
from scipy import stats


######### a function to get integral image

def GetIntegralImage(image):
    r, c = image.shape
    integral = np.zeros((r,c))
    s = np.zeros((c,))
    for i in range(r):
        if i>0:
            for j in range(c):
                if j>0:
                    s[j] = s[j-1] + image[i,j]
                else:
                    s[j] = image[i,j]
                integral[i,j] = integral[i-1,j] + s[j]
        else:
            for j in range(c):
                if j>0:
                    s[j] = s[j-1]+image[i,j]
                else:
                    s[j] = image[i,j]
            integral[i,:] = s
    return(integral)



class ImageReader:
    def __init__(self, top_dir):
        self.top_dir = top_dir

    def __iter__(self):
        for root, dirs, files in os.walk(self.top_dir):
            for fname in filter(lambda fname: fname.endswith('.jpg') and (not fname.startswith('.')), files):
                image = misc.imread(os.path.join(root, fname), flatten=True)
                yield image


class ImageParser:
    def __init__(self, window, stride, image):
        self.window = window
        self.stride = stride
        self.image = image

    def SetWindow(self, window):
        self.window = window

    def SetStride(self, stride):
        self.stride = stride

    def SetImage(self, image):
        self.image = image

    def __iter__(self):
        wd_h, wd_w = self.window
        h, w = self.image.shape
        integral = np.zeros((h + 1, w + 1))
        integral[1:, 1:] = GetIntegralImage(self.image)
        # a = int(np.ceil((h-wd_h+1)/self.stride))
        b = int(np.ceil((w - wd_w + 1) / self.stride))
        parsed = np.zeros(
            (b, wd_h + 1, wd_w + 1))  # include zero-padding on top and on left to facilitate feature selecture
        for i in range(0, h - wd_h + 1, self.stride):
            idx = 0
            for j in range(0, w - wd_w + 1, self.stride):
                parsed[idx, :, :] = integral[i:(i + wd_h + 1), j:(j + wd_w + 1)]
                idx += 1
            yield parsed

def ClusterFace(IsFace):
    r, c = IsFace.shape
    assign = np.zeros((r,c), dtype="int")
    stack = []
    num = 1
    for i in range(r):
        for j in range(c):
            if IsFace[i,j] == True and assign[i,j]==0:
                stack.append([i,j])
                assign[i,j] = num
                print(num)
                num += 1
                while(len(stack)>0):
                    base = stack[-1]
                    ind = 0
                    if i-1>=0 and IsFace[i-1,j]==True and assign[i-1,j]==0:
                        assign[i-1, j] = assign[i,j]
                        ind = 1
                        stack.append([i-1,j])
                    if i+1<=(r-1) and IsFace[i+1,j]==True and assign[i+1,j]==0:
                        assign[i+1, j] = assign[i, j]
                        ind = 1
                        stack.append([i+1, j])
                    if j-1 >= 0 and IsFace[i, j-1] == True and assign[i,j-1]==0:
                        assign[i, j-1] = assign[i, j]
                        ind = 1
                        stack.append([i, j-1])
                    if j+1 <= (c - 1) and IsFace[i, j+1] == True and assign[i,j+1]==0:
                        assign[i, j+1] = assign[i, j]
                        ind = 1
                        stack.append([i, j+1])
                    if i-1>=0 and j-1>=0 and IsFace[i-1,j-1]==True and assign[i-1,j-1]==0:
                        assign[i-1,j-1] = assign[i,j]
                        ind = 1
                        stack.append([i-1,j-1])
                    if i+1<=(r-1) and j-1>=0 and IsFace[i+1,j-1]==True and assign[i+1,j-1]==0:
                        assign[i+1,j-1] = assign[i,j]
                        ind = 1
                        stack.append([i+1,j-1])
                    if i-1>=0 and j+1<=(c-1) and IsFace[i-1,j+1]==True and assign[i-1,j+1]==0:
                        assign[i-1,j+1] = assign[i,j]
                        ind = 1
                        stack.append([i-1,j+1])
                    if i+1<=(r-1) and j+1<=(c-1) and IsFace[i+1,j+1]==True and assign[i+1,j+1]==0:
                        assign[i+1,j+1] = assign[i,j]
                        ind = 1
                        stack.append([i+1,j+1])
                    if j-2>=0 and IsFace[i,j-2]==True and assign[i,j-2]==0:
                        assign[i,j-2] = assign[i,j]
                        ind = 1
                        stack.append([i,j-2])
                    if j+2 <= (c-1) and IsFace[i, j+2] == True and assign[i, j+2] == 0:
                        assign[i, j+2] = assign[i, j]
                        ind = 1
                        stack.append([i, j+2])
                    if i-2>=0 and IsFace[i-2,j]==True and assign[i-2,j]==0:
                        assign[i-2,j] = assign[i,j]
                        ind = 1
                        stack.append([i-2,j])
                    if i+2 <= (r-1) and IsFace[i + 2, j] == True and assign[i + 2, j] == 0:
                        assign[i + 2, j] = assign[i, j]
                        ind = 1
                        stack.append([i + 2, j])
                    if ind == 0:
                        del stack[-1]
    print("end dfs")
    return(assign)

def Convergence(old, new):
    dist = metrics.pairwise_distances(old, new)
    dist = np.diagonal(dist)
    if sum(dist<1e-3)==len(dist):
        return(True)
    else:
        return(False)


def MeanShift(data, h):
    w = np.zeros(data.shape)
    w[:,:] = data[:,:]
    print(data.shape)
    print(w.shape)
    i = 0
    while True:
        print(i)
        dist = metrics.pairwise_distances(w, data)
        print("done dist")
        K = stats.norm.pdf(dist, 0, h)
        w_new = np.matmul(K, data) / np.sum(K, axis=1)
        if not Convergence(w, w_new):
            w = w_new
            i += 1
        else:
            w = w_new
            break
    return(w)


class FaceLabeler:
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def label(self, image, IsFace, d):
        r, c = IsFace.shape
        rowcol = np.ones((2, r, c))
        rowcol[0, :, :] = rowcol[0, :, :] * np.array(range(r)).reshape((r, 1))
        rowcol[1, :, :] = rowcol[1, :, :] * np.array(range(c)).reshape((1, c))

        spoints = np.zeros((np.sum(IsFace), 2))
        spoints[:, 0] = rowcol[0, :, :][IsFace]
        spoints[:, 1] = rowcol[1, :, :][IsFace]

        MS = cluster.MeanShift(24)
        clusters = MS.fit_predict(spoints)

        labeled = np.zeros((self.h, self.w, 3))

        # convert from grayscale to RBG scale
        for i in range(3):
            labeled[:, :, i] = image[:, :]

        # label and draw
        for k in range(max(clusters) + 1):
            one_group = spoints[clusters == i, :]
            if one_group.shape[0] <= 300:
                continue
            center = np.sum(one_group, axis=0) / one_group.shape[0]
            i = int(center[0])
            j = int(center[1])
            labeled[i, j:(j + d - 1), 0] = 255
            labeled[i, j:(j + d - 1), 1] = 0
            labeled[i, j:(j + d - 1), 2] = 0
            labeled[i + d - 1, j:(j + d - 1), 0] = 255
            labeled[i + d - 1, j:(j + d - 1), 1] = 0
            labeled[i + d - 1, j:(j + d - 1), 2] = 0
            labeled[i:(i + d - 1), j, 0] = 255
            labeled[i:(i + d - 1), j, 1] = 0
            labeled[i:(i + d - 1), j, 2] = 0
            labeled[i:(i + d - 1), j + d - 1, 0] = 255
            labeled[i:(i + d - 1), j + d - 1, 1] = 0
            labeled[i:(i + d - 1), j + d - 1, 2] = 0
        return (labeled)

