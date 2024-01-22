import os
import math

import cv2
import numpy as np

import keras
from keras import backend as K

from .hist import Hist

def fit_max(mx):
    y = math.ceil(math.log(mx, 2))
    return y

def to_fx(x, q):
    y = math.trunc(x * (2 ** q))
    return y

def add16(a,b):
    x = a & 65535
    y = b & 65535
    s = (x + y) & 65525
    return s if s < 32768 else s - 65536

def mod16(a):
    x = a & 65535
    return x if x < 32768 else x - 65536

def quantize_mat(A, frac):
    s = 2 ** frac
    return np.trunc(A * s).astype(np.int32)

def quantize_w_4d_per_feat(w, wbits):
    qw = []
    iwts = []
    for i in range(w.shape[3]):
        mx = np.max(np.abs(w[:,:,:,i]))
        int_bits = fit_max(mx)
        q = (wbits-1)-int_bits
        qw.append(q)
        s = 2 ** q
        sw = np.trunc(w[:,:,:,i:i+1] * s).astype(np.int32)
        iwts.append(sw)
    return (np.concatenate(iwts, axis=-1), qw)

def get_bn_param(bn):
    s1 = bn.gamma / np.sqrt(bn.variance + bn.epsilon)
    s2 = bn.beta - bn.mean * s1
    s3 = np.vstack((s1, s2)).T
    return s3

def fuse_batchnorm_dense(bn_lyr, dense_lyr):
    bn = get_bn_param(bn_lyr)
    W0 = dense_lyr.W
    b0 = conv2d_lyr.b
    W1 = np.multiply(W0, bn[:, 0]).T
    b1 = b0 + np.matmul(np.transpose(bn[:, 1]), W0)
    return [W1, b1]

def fuse_batchnorm_conv2d(bn_lyr, conv2d_lyr):
    bn = get_bn_param(bn_lyr)
    W0 = conv2d_lyr.W
    b0 = conv2d_lyr.b
    W1 = np.multiply(W0, np.tile(bn[:,0], (W0.shape[3],1)).T)
    b1 = b0 + np.matmul(np.sum(W0, axis=(0,1)).T, bn[:,1].T)
    return [W1, b1]

def export_cpp_conv2d(bits, W, name, fout):
    N = W.shape[0]
    if W.shape[1] != N:
        raise Exception('Non-square convolution kernel not implemented')
    M = W.shape[2]
    O = W.shape[3]
    print('const Vec<Vol<int%d_t,%d,%d,%d>,%d> %s({' % (bits,N,N,M,O,name),
          file=fout)
    for fo in range(O):
        print('{', end='', file=fout)
        for y in range(N):
            print('{', end='', file=fout)
            for x in range(N):
                vec = ','.join([str(v) for v in W[y,x,:,fo].tolist()])
                print('{' + vec + '}', end='', file=fout)
                if x < N-1:
                    if x > 0 and x%50 == 0:
                        print(',', file=fout)
                    else:
                        print(',', end='', file=fout)
            if y < N-1:
                print('},', file=fout)
            elif fo < O-1:
                print('}},', file=fout)
            else:
                print('}}});', file=fout)
    print('', file=fout)


def export_cpp_bias(bits, bias, name, fout):
    bias = bias.tolist()
    print('const Vec<int%d_t,%d> %s({' % (bits,len(bias),name),
          file=fout)
    vec = ','.join([str(v) for v in bias])
    print(vec + '});', file=fout)
    print('', file=fout)
    

# output type same as kernel type
def convolve2D(image, kernel, strides=(1,1)):
    ky = kernel.shape[0]
    kx = kernel.shape[1]
    NI = kernel.shape[2]
    NO = kernel.shape[3]
    yres = image.shape[0]
    xres = image.shape[1]

    ypad = ky-strides[0]
    xpad = kx-strides[1]
    ypad0 = ypad//2
    xpad0 = xpad//2
    padded = np.zeros((yres+ypad, xres+xpad, NI), dtype=kernel.dtype)
    padded[ypad0:ypad0+yres,xpad0:xpad0+xres,:] = image

    xres_o = int(((xres - kx + xpad)//strides[1]) + 1)
    yres_o = int(((yres - ky + ypad)//strides[0]) + 1)

    out = np.zeros((yres_o, xres_o, NO), dtype=kernel.dtype)
    for y in range(yres_o):
        py = y*strides[0]
        for x in range(xres_o):
            px = x*strides[1]
            out[y, x, :] = np.tensordot(padded[py:py+ky, px:px+kx], kernel, 3)

    return out

class NpLayer:
    def __init__(self):
        self.track = False

class NpInput(NpLayer):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, X):
        return X

class NpNop(NpLayer):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, X):
        return X

class NpRelu(NpLayer):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return np.maximum(0, X)

class NpConv2D(NpLayer):
    def __init__(self, strides, weights, bias):
        super().__init__()
        self.strides = strides;
        self.W = weights
        self.b = bias
        self.NO = weights.shape[3]
        self.history = [Hist() for i in range(self.NO)]

    def forward(self, X):
        X = convolve2D(X, self.W, self.strides) + self.b
        if self.track:
            for fo in range(self.NO):
                self.history[fo].add(X[:,:,fo].flatten())
        return X

class NpBatchNorm(NpLayer):
    def __init__(self, gamma, beta, mean, variance, epsilon):
        super().__init__()
        self.gamma = gamma
        self.beta = beta
        self.mean = mean
        self.variance = variance
        self.epsilon = epsilon
        self.scale = gamma/np.sqrt(variance + epsilon)

    def forward(self, X):
        return self.scale * (X - self.mean) + self.beta

class NpSeq(NpLayer):
    def __init__(self, seq):
        super().__init__()
        self.seq = seq

    def forward(self, X):
        for idx,lyr in enumerate(self.seq):
            X = lyr.forward(X)
        return X

# Quantized Conv2D with per feature (output) scale
class NpFtQuantConv2D(NpLayer):
    def __init__(self, strides, weights, bias, qw, preshr, qb, postshr, qi, qo):
        super().__init__()
        self.strides = strides;
        self.W = weights
        self.b = bias
        self.qw = qw
        self.qb = qb
        self.preshr = preshr
        self.postshr = postshr
        self.qi = qi
        self.qo = qo

    def forward(self, X):
        acc = convolve2D(X, self.W, self.strides)
        acc = np.right_shift(acc, self.preshr)
        for o in range(len(self.b)):
            for y in range(acc.shape[0]):
                for x in range(acc.shape[1]):
                    acc[y,x,o] = add16(acc[y,x,o], self.b[o]) >> self.postshr[o]
        return acc

class NpLyQuantConv2D(NpLayer):
    def __init__(self, strides, weights, bias, bshl, ashr, qi, qo):
        super().__init__()
        self.strides = strides;
        self.W = weights
        self.b = bias
        self.bshl = bshl
        self.ashr = ashr
        self.qi = qi
        self.qo = qo

    def forward(self, X):
        acc = convolve2D(X, self.W, self.strides)
        bias = self.b * (2**self.bshl)
        acc = np.right_shift(acc + bias, self.ashr)
        return np.clip(acc, -(2**15), 2**25-1)

class QuantParam:
    def __init__(self, qi):
        self.qi = qi
        self.per_feature = False

def quantize_conv2d(lyr, param, wbits, debug):
    qi = param.qi
    n_out = lyr.W.shape[3]

    ########## not currently used
    # per feature scale is only supported on torte engine /w dense-only nets
    if param.per_feature:
        (iwts, qw) = quantize_w_4d_per_feat(lyr.W, wbits)

        # per-output product fractional bits
        qp = [qi + q for q in qw]

        # per-output maximum accumulator fractional bits
    #            bias_mag = [fit_max(abs(x)) for x in rbias]
        act_mag = [fit_max(h.absmax()) for h in lyr.history]
    #            qa_max = [15-max(bias_mag[i], act_mag[i]) for i in range(len(rbias))]
        qa_max = [15-act_mag[o] for o in range(n_out)]
        if any([x < 0 for x in qa_max]):
            raise Exception('activation overflow')

        # per-output required product shift right
        if any([qp[o]<qa_max[o] for o in range(len(qp))]):
            raise Exception('negative post bias shift')
        shr = [qp[o]-qa_max[o] for o in range(len(qp))]

        # per-layer product shift right
        preshr = max(shr)

        # per-output post-shift product fractional bits
        qa = [q - preshr for q in qp]
        amax = max(h.absmax() for h in lyr.history)
        if debug:
            print(amax, preshr,'=>',qa)

        # per-output bias fractional bits
        qb = qa
        # qb is based on the final activation value, so bias values may overflow the integer
        # bits.  However, this does not matter as long as the sum is in range then act + mod16(bias)
        # still be correct.
        ibias = np.array([mod16(to_fx(lyr.b[i], qb[i])) for i in range(len(qb))])

        # per-layer output fractional bits
        qo = min(qa)

        # per-output activation shift right
        postshr = [q - qo for q in qa]

        if debug:
            print(preshr, postshr, qo)

        param.qi = qo
        return NpFtQuantConv2D(lyr.strides, iwts, ibias, qw, preshr, qb, postshr, qi, qo)
    else:
        # per layer scale
        bbits = 16

        # max possible frac bits for weight matrix
        mx = np.max(np.abs(lyr.W))
        max_qw = (wbits-1)-fit_max(mx)
        if max_qw < 0:
            raise Exception('Cannot quantize weights to %d bits' % (wbits,))
        qw = max_qw
        prod_q = qi + qw

        # max possible frac bits for bias vector
        bias_mag = [fit_max(abs(x)) for x in lyr.b]
        max_qb = (bbits-1)-max(bias_mag)
        if max_qb < 0:
            raise Exception('Cannot quantize weights to %d bits' % (bbits,))

        if prod_q > max_qb:
            qb = max_qb
            bshl = prod_q - qb
        else:
            qb = prod_q
            bshl = 0

        # max possible frac bits for activation matrix
        act_mag = [fit_max(h.absmax()) for h in lyr.history]
        max_qa = 15-max(act_mag)
        if max_qa < 0:
            raise Exception('Activation value overflow')
        if prod_q > max_qa:
            qa = max_qa
            ashr = prod_q - max_qa
        else:
            qa = prod_q
            ashr = 0
        qo = qa

        if debug:
            print('%d (%d, %d) -> %d' % (qi, qw, qb, qo))

        iwts = quantize_mat(lyr.W, qw)
        ibias = quantize_mat(lyr.b, qb)

        param.qi = qo
        return NpLyQuantConv2D(lyr.strides, iwts, ibias, bshl, ashr, qi, qo)

def keras_to_np_layer(lyr, net):
    cls = lyr.__class__.__name__
    cfg = lyr.get_config()
    if cls == 'InputLayer':
        # first element is batch size
        net.append(NpInput(cfg['batch_input_shape'][1:]))
    elif cls == 'Conv2D':
        if cfg['padding'] != 'same':
            raise Exception(cfg['padding'] + ' padding not implemented')
        if cfg['dilation_rate'] != (1,1):
            raise Exception('dialation not implemented')
        if not cfg['use_bias']:
            raise Exception('expecting bias')

        [weights, bias] = lyr.weights
        conv = NpConv2D(cfg['strides'], weights.numpy(), bias.numpy())
        if cfg['activation'] == 'relu':
            net.append(NpSeq([conv, NpRelu()]))
        elif cfg['activation'] == 'linear':
            net.append(conv)
        else:
            raise Exception('Unimplemented activation: '+cfg['activation'])
    elif cls == 'BatchNormalization':
        W = [w.numpy() for w in lyr.weights]
        net.append(NpBatchNorm(W[0], W[1], W[2], W[3], cfg['epsilon']))
    elif cls == 'Dropout':
        net.append(NpNop())
    else:
        print(cfg)
        if cfg.get('trainable', False):
            print([w.name for w in lyr.weights])
        raise Exception('Unimplemented layer: '+cls)

def debug_layer(model, target, X):
    npo = X
    for idx,lyr in enumerate(net):
        npo = lyr.forward(npo)
        if idx == target:
            break

    fun = K.function([model.layers[0].input], [model.layers[target].output])
    ko = fun(np.expand_dims(X, axis=0))[0]

    print(npo[0,0,:])
    print(ko[0,0,0,:])
    print(np.allclose(ko[0,:,:,:],npo,atol=1e-3))

def listall(layers, indent='  '):
    for idx in range(len(layers)):
        lyr = layers[idx]
        print(indent + lyr.__class__.__name__)
        if isinstance(lyr, NpSeq):
            listall(lyr.seq, indent+'  ')

def flatten_layers(layers):
    net = []
    for idx in range(len(layers)):
        lyr = layers[idx]
        if isinstance(lyr, NpSeq):
            net.extend(lyr.seq)
        elif isinstance(lyr, NpNop):
            pass
        elif isinstance(lyr, NpInput):
            pass
        else:
            net.append(lyr)
    return net

def merge_layers(layers):
    net = []
    for idx in range(len(layers)):
        lyr = layers[idx]
        if idx > 0 and isinstance(lyr, NpConv2D) and isinstance(net[-1], NpBatchNorm):
            [W,b] = fuse_batchnorm_conv2d(net.pop(), lyr)
            net.append(NpConv2D(lyr.strides, W, b))
        else:
            net.append(lyr)
    return net


class QNNet:
    def __init__(self, layers, nc, wbits, qi, qo):
        self.layers = layers
        self.nc = nc
        self.qi = qi
        self.qo = qo
        self.wbits = wbits

    def summary(self):
        listall(self.layers)

    def predict(self, X):
        X = np.right_shift(X.astype(np.int32), 1)
        for lyr in self.layers:
            X = lyr.forward(X)
        s = 2**self.qo
        return X.astype(np.float32) / s

    # Undocumented method used for internal testing
    def export_cpp(self, basename, cout, hout):
        nc = self.nc
        print('#include "sximage.h"', file=cout)
        print('', file=cout)

        if nc == 1:
            print('ImageIn<Gray> src("your test image");', file=cout)
        else:
            print('ImageIn<Gray> src("your test image");', file=cout)
        print('ImageOut<Gray> out("%s.png");' % (basename,), file=cout)
        print('', file=cout)

        print('int main(int argc, char *argv[]) {', file=cout)
        if nc == 1:
            print('  src >> CnnInputGr<Gray>() >>', file=cout)
        elif nc == 3:
            print('  src >> CnnInputRgb<Rgb>() >>', file=cout)
        else:
            raise Exception('Unexpected number of input channels: '+str(nc))
        bits = 8
        frac = 7
        idx = 0
        while idx < len(self.layers):
            positive = False
            lyr = self.layers[idx]
            if isinstance(lyr, NpLyQuantConv2D):
                nc = lyr.W.shape[3]
                export_cpp_conv2d(self.wbits, lyr.W, '_conv' + str(idx), hout)
                export_cpp_bias(16, lyr.b, '_bias' + str(idx), hout)
                print('    CnnConv<int%d_t,int32_t>(_conv%d) >>' % (bits,idx), file=cout)
                nxt = idx+1
                if nxt < len(self.layers) and isinstance(self.layers[nxt], NpRelu):
                    print('    CnnBiasRelu<int32_t,int16_t>(_bias%d, %d, %d) >>' % (idx,lyr.bshl,lyr.ashr), file=cout)
                    idx = nxt
                    positive = True
                else:
                    print('    CnnBias<int32_t>(_bias%d, %d) >>' % (bits,idx,lyr.bshl), file=cout)
                    print('    CnnReduce<int32_t,int16_t>(%d) >>' % (lyr.ashr), file=cout)  
                bits = 16
                frac = lyr.qo
            elif isinstance(lyr, NpRelu):
                print('    CnnRelu<int%d_t,%d>() >>' % (bits,nc), file=cout)
                positive = True
            else:
                raise Exception('Cannot export layer: ' + lyr.__class__.__name__)
            idx += 1
        if not positive:
            print('    CnnRelu<int%d_t,%d>() >>' % (bits,nc), file=cout)
        shr = 8 if frac>8 else frac
        frac -= shr
        print('    CnnReduce<int16_t,uint8_t>(%d) >>' % (shr), file=cout)
        print('    out;', file=cout)
        print('  return 0;', file=cout)
        print('}', file=cout)
        print('WARNING: final output is %d.%d unsigned fixed point' % (8-frac,frac))
    def export(self, filename):
        basename = os.path.splitext(os.path.basename(filename))[0]
        fout = open(filename, 'w')
        if filename.endswith('.cpp'):
            hout = open(os.path.splitext(filename)[0] + '.h', 'w')
            self.export_cpp(basename, fout, hout)

class NNet:
    def __init__(self, layers):
        self.layers = layers

    def summary(self):
        listall(self.layers)

    def simplify(self):
        self.layers = flatten_layers(self.layers)
        self.layers = merge_layers(self.layers)

    def predict(self, X):
        for lyr in self.layers:
            X = lyr.forward(X)
        return X

    def quantize(self, files, wbits=8, debug=False):
        for lyr in self.layers:
            lyr.track = True
                
        nf = len(files)
        print('Analyzed %d of %d' % (0,nf), end='')
        for idx,filename in enumerate(files):
            img = cv2.imread(filename)
            X = img.astype(np.float32) / 256
            self.predict(X)
            if idx % 10 == 0:
                print('\rAnalyzed %d of %d' % (idx,nf), end='')
        print('\rAnalyzed %d of %d' % (nf,nf))

        nc = img.shape[2]
        q_in = 7
        qo = 7
        param = QuantParam(q_in)
        net = []
        for idx in range(len(self.layers)):
            lyr = self.layers[idx]
            if isinstance(lyr, NpConv2D):
                conv = quantize_conv2d(lyr, param, wbits, debug)
                net.append(conv)
                qo = conv.qo
            else:
                net.append(lyr)
        if not isinstance(net[-1], NpRelu):
            net.append(NpRelu())

        return QNNet(net, nc, wbits, q_in, qo)

def load_from_keras(model):
    net = []
    for lyr in model.layers:
        keras_to_np_layer(lyr, net)
    return NNet(net)
