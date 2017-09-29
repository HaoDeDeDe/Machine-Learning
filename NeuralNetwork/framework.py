import numpy as np

DataType = np.float32

parameters = []
nodes = []

def Forward():
    for node in nodes:
        node.forward()

def Backward(loss):
    for node in nodes:
        if node.gradient is not None:
            node.gradient = DataType(0)
    loss.gradient = np.ones_like(loss.value)
    for node in nodes[::-1]:
        node.backward()

def SGD(eta):
    for para in parameters:
        para.value = para.value - eta * para.gradient
        para.gradient = DataType(0)


b1t = DataType(1.0)
b2t = DataType(1.0)
def Adam(eta=0.001,b1=0.9,b2=0.999,eps=1e-8):
    global b1t
    global b2t

    if 'mt' not in parameters[0].__dict__.keys():
        for p in parameters:
            p.mt = DataType(0)
            p.vt = DataType(0)

    b1 = DataType(b1)
    b2 = DataType(b2)
    eps = DataType(eps)
    b1t = b1t * b1
    b2t = b2t * b2
    for p in parameters:
        p.mt = b1 * p.mt + (1.-b1) * p.gradient
        p.vt = b2 * p.vt + (1.-b2) * p.gradient * p.gradient
        mhat = p.mt / (1. - b1t)
        vhat = p.vt / (1. - b2t)
        p.gradient = mhat / (np.sqrt(vhat) + eps)

    SGD(eta)

def predict():
    for node in nodes:
        if type(node).__name__ == 'SquaredLoss':
            break
        node.forward()

def accuracy(yhat, y):
    yhat = np.argmax(yhat, axis=1)
    y = np.argmax(y, axis=1)
    n = y.shape[0]
    accur = np.sum(np.equal(yhat,y)) / float(n)
    return(accur)


class Para:
    def __init__(self, shape):
        sq = np.sqrt(3.0 / np.prod(shape[:-1]))
        self.value = np.random.uniform(-sq,sq,shape)   # xavier initialization
        self.gradient = DataType(0)
        parameters.append(self)


class Value:
    def __init__(self, value, shape):
        self.value = DataType(value).copy()
        self.shape = shape
        self.gradient = None

    def set(self, value, shape):
        self.value = DataType(value).copy().reshape(shape)
        self.shape = shape

'''
Class: VAdd
Usage: matrix or vector addition, supports batch-training by broadcasting, but if train example-by-example, broadcasting won't be needed
Arguments: a can be a matrix or a vector, b is usually a bias vector of NN
'''
class VAdd:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.value = DataType(0)
        if self.a.gradient is None and self.b.gradient is None:
            self.gradient = None
        else:
            self.gradient = DataType(0)
        nodes.append(self)

    def forward(self):
        self.value = self.a.value + self.b.value

    def backward(self):
        if self.a.gradient is not None:
            self.a.gradient = self.a.gradient + self.gradient
        if self.b.gradient is not None:
            self.b.gradient = self.b.gradient + np.sum(self.gradient.reshape([-1, len(self.b.value)]), axis=0)

'''
Class: VMul
Usage: matrix or vector multiplication
Arguments: a can be a matrix or a vector, b is usually a matrix (a weight matrix of NN)
'''
class VMul:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.value = DataType(0)
        if self.a.gradient is None and self.b.gradient is None:
            self.gradient = None
        else:
            self.gradient = DataType(0)
        nodes.append(self)

    def forward(self):
        self.value = np.dot(self.a.value, self.b.value)

    def backward(self):
        if self.a.gradient is not None:
            self.a.gradient = self.a.gradient + np.dot(self.gradient, self.b.value.T)
        if self.b.gradient is not None:
            self.b.gradient = self.b.gradient + np.dot(self.a.value.T, self.gradient)

'''
Class: Mean
Usage: compute mean of a tensor
Arguments: x can be any tensor, but usually a vector
'''
class Mean:
    def __init__(self, x):
        self.x = x
        self.value = None
        if self.x.gradient is None:
            self.gradient = None
        else:
            self.gradient = DataType(0)
            self.gradient = DataType(0)
        nodes.append(self)

    def forward(self):
        self.value = np.mean(self.x.value)

    def backward(self):
        if self.x.gradient is not None:
            self.x.gradient = self.x.gradient + self.gradient * np.ones_like(self.x.value) / np.prod(self.x.value.shape)

'''
Class: Sigmoid
Usage: element-wise sigmoid transfer
Arguments: x can be numpy arrays of any shape
'''
class Sigmoid:
    def __init__(self, x):
        self.x = x
        self.value = DataType(0)
        if self.x.gradient is None:
            self.gradient = None
        else:
            self.gradient = DataType(0)
        nodes.append(self)

    def forward(self):
        self.value = np.exp(self.x.value) / ( 1 + np.exp(self.x.value) )

    def backward(self):
        if self.x.gradient is not None:
            self.x.gradient = self.x.gradient + self.gradient * self.value * ( 1. - self.value )

'''
Class: SquaredLoss
Usage: compute squared loss
Arguments: yhat and y are both vectors or matrices of the same shape; yhat contains predicted values and y contains true values
'''
class SquaredLoss:
    def __init__(self, yhat, y):
        self.yhat = yhat
        self.y = y
        self.value = DataType(0)
        if self.yhat.gradient is None and self.y.gradient is None:
            self.gradient = None
        else:
            self.gradient = np.float32(0)
        nodes.append(self)

    def forward(self):
        diff = self.yhat.value - self.y.value
        self.value = np.sum(diff * diff, axis=1, keepdims=True)

    def backward(self):
        if self.yhat.gradient is not None:
            self.yhat.gradient = self.yhat.gradient + 2 * (self.yhat.value-self.y.value) * self.gradient
        if self.y.gradient is not None:
            self.y.gradient = self.y.gradient + 2 * (self.y.value-self.yhat.value) * self.gradient


'''
Class: Softmax
Usage: Compute Softmax transfer function, supports batch training, uses max to rescale and avoid overflow
Arguments: x can be a matrix or a vector, if batch training is used, x will be a matrix
'''
class Softmax:
    def __init__(self, x):
        self.x = x
        if self.x.gradient is None:
            self.gradient = None
        else:
            self.gradient = DataType(0)
        nodes.append(self)

    def forward(self):
        lmax = np.max(self.x.value, axis=-1, keepdims=True)
        expo = np.exp( self.x.value - lmax )
        self.value = expo / np.sum(expo, axis=-1, keepdims=True)

    def backward(self):
        if self.x.gradient is not None:
            self.x.gradient = self.x.gradient + self.value * ( self.gradient - np.sum(self.gradient*self.value, axis=-1, keepdims=True) )


'''
Class: Conv
Usage: Convolution layer, supports multi-channel convolution and batch training
Arguments: im represents image, and is a tensor with 4 dims, dim[0] is for batch, dim[1] and dim[2] are image height and width respectively, dim[3] is for channel
                i.e., im has shape (B, H, W, C)
           flt represents a square filter, and is a tensor with 4 dims, dim[0] and dim[1] are filter height and width, dim[2] is same as is the input channel, dim[3] is the output channel
                i.e. flt has shape (K, K, C, C')
'''
class Conv:
    def __init__(self,im, flt, stride, pad):
        self.im = im
        self.flt = flt
        self.stride = stride
        self.value = DataType(0)
        pad = np.array(pad)
        if pad.shape == ():
            self.xpad = self.ypad = pad
        else:
            self.xpad = pad[1]
            self.ypad = pad[0]
        if self.im.gradient is None and self.flt.gradient is None:
            self.gradient = None
        else:
            self.gradient = DataType(0)
        nodes.append(self)

    def forward(self):
        B = self.im.value.shape[0]
        H = self.im.value.shape[1]
        W = self.im.value.shape[2]
        C = self.im.value.shape[3]
        K = self.flt.value.shape[0]
        Ct = self.flt.value.shape[3]

        # padding
        self.padded = np.zeros(( B, (H+2*self.ypad), (W+2*self.xpad), C ))
        self.padded[:,self.ypad:(H+self.ypad), self.xpad:(W+self.xpad), :] = self.im.value

        # convolve
        self.value = np.zeros(( B, np.int32((H+2*self.ypad-K)/self.stride)+1, np.int32((W+2*self.xpad-K)/self.stride)+1, Ct ))
        flt_reshaped = self.flt.value.reshape((K*K*C, Ct))
        for i in range(self.value.shape[1]):
            for j in range(self.value.shape[2]):
                tmp = self.padded[:, (i*self.stride):(i*self.stride+K), (j*self.stride):(j*self.stride+K), :].reshape((B,K*K*C))
                self.value[:,i,j,:] = np.matmul(tmp, flt_reshaped)

    def backward(self):
        H = self.im.value.shape[1]
        W = self.im.value.shape[2]
        K = self.flt.value.shape[0]

        padded_T = np.transpose(self.padded, (1,2,3,0))
        flt_T = np.transpose(self.flt.value, (0,1,3,2))
        padded_g = np.zeros(self.padded.shape)
        flt_g = np.zeros(self.flt.value.shape)

        if self.gradient is not None:
            for i in range(self.value.shape[1]):
                for j in range(self.value.shape[2]):
                    padded_g[:, (i*self.stride):(i*self.stride+K), (j*self.stride):(j*self.stride+K), :] += np.dot(self.gradient[:,i,j,:], flt_T)
                    flt_g += np.dot(padded_T[(i*self.stride):(i*self.stride+K), (j*self.stride):(j*self.stride+K),:,:], self.gradient[:,i,j,:])

        if self.im.gradient is not None:
            self.im.gradient = self.im.gradient + padded_g[:,self.ypad:(H+self.ypad), self.xpad:(W+self.xpad), :]

        if self.flt.gradient is not None:
            self.flt.gradient = self.flt.gradient + flt_g


'''
Class: Maxpool
Usage: max pool sampling, supports batch training and batch prediction
Arguments: im represents image, d is an integer that represents the patch size, stride is an integer
'''
class Maxpool:
    def __init__(self, im, d, stride):
        self.im = im
        self.d = d
        if stride is None:
            self.stride = self.d
        else:
            self.stride = stride
        if self.im.gradient is None:
            self.gradient = None
        else:
            self.gradient = DataType(0)
        nodes.append(self)

    def forward(self):
        self.value = -1 * float("Inf")
        for i in range(self.d):
            for j in range(self.d):
                self.value = np.maximum(self.value, self.im.value[:, i::self.stride, j::self.stride, :] )

    def backward(self):
        if self.im.gradient is not None:
            self.im.gradient = np.zeros(self.im.value.shape)
            for i in range(self.d):
                for j in range(self.d):
                    self.im.gradient[:, i::self.stride, j::self.stride, :] = self.im.gradient[:, i::self.stride, j::self.stride, :] \
                                                                             + self.gradient * np.equal(self.value, self.im.value[:, i::self.stride, j::self.stride, :])

'''
Class: Avgpool
Usage: average pool sampling, supports batch training and batch prediction
Arguments: im represents image, d is an integer that represents the patch size, stride is an integer
'''
class Avgpool:
    def __init__(self, im, d, stride):
        self.im = im
        self.d = d
        self.value = DataType(0)
        if stride is None:
            self.stride = self.d
        else:
            self.stride = stride
        if self.im.gradient is None:
            self.gradient = None
        else:
            self.gradient = DataType(0)
        nodes.append(self)

    def forward(self):
        self.value = DataType(0)
        for i in range(self.d):
            for j in range(self.d):
                self.value = self.value + self.im.value[:, i::self.stride, j::self.stride, :]
        self.value = self.value / DataType(self.d) / DataType(self.d)

    def backward(self):
        if self.im.gradient is not None:
            self.im.gradient = np.zeros(self.im.value.shape)
            avg = 1 / DataType(self.d) / DataType(self.d)
            for i in range(self.d):
                for j in range(self.d):
                    self.im.gradient[:, i::self.stride, j::self.stride, :] = self.im.gradient[:, i::self.stride, j::self.stride, :] \
                                                                             + avg * self.gradient

'''
Class: Reshape
Usage: reshape the value of an object to a particular shape
'''
class Reshape:
    def __init__(self, x, shape):
        self.x = x
        self.shape = shape
        self.value = DataType(0)
        if self.x.gradient is None:
            self.gradient = None
        else:
            self.gradient = DataType(0)
        nodes.append(self)

    def update(self, shape):
        self.shape = shape

    def forward(self):
        self.value = np.reshape(self.x.value, self.shape)

    def backward(self):
        if self.x.gradient is not None:
            self.x.gradient = self.x.gradient + np.reshape(self.gradient, self.x.value.shape)

















