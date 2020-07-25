
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Input, Conv2DTranspose
from tensorflow.keras.layers import concatenate, add, UpSampling2D, ReLU, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K

class name(Layer):    
    def __init__(self, **kwargs):
        super(name, self).__init__()
        pass
    
    def call(self, inputs, training):
        pass

class conv_block(Layer):
    def __init__(self, ch_out = 32, kernel = 3, bn = True, drp = True, res = True):
        super(conv_block, self).__init__()
        
        self.bn = bn
        self.drp = drp
        self.res = res
        self.conv1 = Conv2D(ch_out, kernel, padding = 'same')
        self.conv2 = Conv2D(ch_out, kernel, padding = 'same')
        if self.bn:
            self.bn1 = BatchNormalization()
            self.bn2 = BatchNormalization()
        if self.drp:
            self.drp1 = Dropout(0.5)
            self.drp2 = Dropout(0.5)
        if self.res:
            self.conv1x1 = Conv2D(ch_out, 1 , padding = 'same')
            
    def call(self, inputs, training):
        x = inputs
        x = self.conv1(x)
        if self.bn:
            x = self.bn1(x, training = training)
        x = ReLU()(x)
        if self.drp:
            x = self.drp1(x, training = training)
        x = self.conv2(x)
        if self.bn:
            x = self.bn1(x, training = training)
        x = ReLU()(x)
        if self.drp:
            x = self.drp2(x, training = training)
        return (x if not self.res else x + self.conv1x1(inputs))
    
class rec_layer(Layer):
    def __init__(self, ch_out = 32, kernel = 3, bn = True, t = 2):
        super(rec_layer, self).__init__()
        self.t = t
        self.bn = bn
        self.relu = ReLU()
        self.conv = Conv2D(ch_out, kernel, padding = 'same')
        if self.bn:
            self.bn1 = BatchNormalization()

    def call(self, inputs, training):
        for i in range(self.t):
            if i == 0:
                x = self.conv(inputs)
                if self.bn:
                    x = self.bn1(x, training)
                x = self.relu(x)
                
            x = self.conv(add([x, inputs]))
            if self.bn:
                x = self.bn1(x, training)
            x = self.relu(x)
        return x
    def my_func(self, input_shape, training):
        inputs1 = Input(input_shape)
        inputs = inputs1
        for i in range(self.t):
            if i == 0:
                x = self.conv(inputs)
                if self.bn:
                    x = self.bn1(x, training)
                x = self.relu(x)
                
            x = self.conv(add([x, inputs]))
            if self.bn:
                x = self.bn1(x, training)
            x = self.relu(x)
        model = Model(inputs1, x)
        plot_model(model)
        print(model.summary())


class rec_block(Layer):
    def __init__(self, ch_out = 32, kernel = 3, t = 2, bn = True, drp = True, res = True):
        super(rec_block, self).__init__()
        self.t = t
        self.bn = bn
        self.drp = drp
        self.res = res
        self.rec1 = rec_layer(ch_out, kernel, self.bn, self.t)
        self.rec2 = rec_layer(ch_out, kernel, self.bn, self.t)
        self.conv1x1 = Conv2D(ch_out, 1, padding = 'same')
        if self.drp:
            self.drp1 = Dropout(0.5)

    def call(self, inputs, training):
        inputs = self.conv1x1(inputs)
        x = self.rec1(inputs, training)
        x = self.rec2(x, training)
        if self.drp:
            x = self.drp1(x, training)
        return x if not self.res else x + inputs

    def my_func(self, input_shape, training = False):
        inputs = Input(input_shape)
        x = self.conv1x1(inputs)
        x = self.rec1(x, training)
        x = self.rec2(x, training)
        if self.drp:
            x = self.drp1(x, training)
        x = x if not self.res else x + inputs
        model = Model(inputs, x)
        plot_model(model)
        print(model.summary())
        

class de_conv(Layer):
    def __init__(self, ch_out, kernel = 3, bn = True):
        super(de_conv, self).__init__()
        self.bn = bn
        self.deconv = Conv2DTranspose(ch_out, kernel, strides = 2, padding = 'same')
        if self.bn:
            self.bn1 = BatchNormalization()
        pass
    
    def call(self, inputs, training):
        inputs = self.deconv(inputs)
        if self.bn:
            inputs = self.bn1(inputs, training)
        inputs = ReLU()(inputs)
        return inputs
    
class UNET(Model):
    def __init__(self, depth = 4, ch_init = 32, kernel = 3, map_ = 2, sig = True, bn = True, drp = False, res = True):
        super(UNET, self).__init__()
        
        self.bn = bn
        self.drp = drp
        self.res = res
        self.sig = sig
        self.d = depth
        self.conv_down = []
        self.deconv = []
        self.conv_up = []
        self.pool = []
        self.map = map_ 
        for i in range(depth):
            self.pool.append(MaxPool2D(2,2))
        
        for i in range(depth + 1):
            self.conv_down.append(conv_block(ch_init*(2**i), kernel, bn, drp, res))
        
        for i in range(depth):
            self.conv_up.append(conv_block(ch_init*(2**i), kernel, bn, drp, res))
        
        for i in range(depth):
            self.deconv.append(de_conv(ch_init*(2**i), kernel, bn))
        
        self.last = Conv2D(map_, kernel, padding = 'same', activation = 'relu')
   
    def call(self, inputs, training):
        skips = []
        for i in range(self.d):
            inputs = self.conv_down[i](inputs, training)
            skips.append(inputs)
            inputs = self.pool[i](inputs)
        
        inputs = self.conv_down[self.d](inputs)
        
        for i in range(self.d):
            inputs = self.deconv[self.d - 1 - i](inputs, training)
            inputs = concatenate([skips[self.d - 1 - i], inputs], axis = 3)
            inputs = self.conv_up[self.d - 1 - i](inputs, training)
        
        inputs = self.last(inputs)
        
        if self.sig:
            inputs = sigmoid(inputs)
        
        return inputs

    def my_func(self, input_shape, training):
        ins = Input(input_shape)
        inputs = ins
        skips= []
        for i in range(self.d):
            inputs = self.conv_down[i](inputs, training)
            skips.append(inputs)
            inputs = self.pool[i](inputs)
        
        inputs = self.conv_down[self.d](inputs)
        
        for i in range(self.d):
            inputs = self.deconv[self.d - 1 - i](inputs, training)
            inputs = concatenate([skips[self.d - 1 - i], inputs], axis = 3)
            inputs = self.conv_up[self.d - 1 - i](inputs, training)
        
        inputs = self.map(inputs)
        
        if self.sig:
            inputs = sigmoid(inputs)
        model = Model(ins, inputs)
        plot_model(model)


class R2UNET(Model):
    def __init__(self, depth = 4, ch_init = 32, kernel = 3, t = 2, map_ = 2, sig = True, bn = True, drp = False, res = True):
        super(R2UNET, self).__init__()
        
        self.bn = bn
        self.drp = drp
        self.res = res
        self.sig = sig
        self.d = depth
        self.t = t
        self.conv_down = []
        self.deconv = []
        self.conv_up = []
        self.pool = []
        self.map = map_
        for i in range(depth):
            self.pool.append(MaxPool2D(2,2))
        
        for i in range(depth + 1):
            self.conv_down.append(rec_block(ch_init*(2**i), kernel, t, bn, drp, res))
        
        for i in range(depth):
            self.conv_up.append(rec_block(ch_init*(2**i), kernel, t, bn, drp, res))
        
        for i in range(depth):
            self.deconv.append(de_conv(ch_init*(2**i), kernel, bn))
        
        self.last = Conv2D(map_, kernel, padding = 'same', activation = 'relu')
   
    def call(self, inputs, training):
        skips = []
        for i in range(self.d):
            inputs = self.conv_down[i](inputs, training)
            skips.append(inputs)
            inputs = self.pool[i](inputs)
        
        inputs = self.conv_down[self.d](inputs)
        
        for i in range(self.d):
            inputs = self.deconv[self.d - 1 - i](inputs, training)
            inputs = concatenate([skips[self.d - 1 - i], inputs], axis = 3)
            inputs = self.conv_up[self.d - 1 - i](inputs, training)
        
        inputs = self.last(inputs)
        
        if self.sig:
            inputs = sigmoid(inputs)
        
        return inputs

    def my_func(self, input_shape, training):
        ins = Input(input_shape)
        inputs = ins
        skips= []
        for i in range(self.d):
            inputs = self.conv_down[i](inputs, training)
            skips.append(inputs)
            inputs = self.pool[i](inputs)
        
        inputs = self.conv_down[self.d](inputs)
        
        for i in range(self.d):
            inputs = self.deconv[self.d - 1 - i](inputs, training)
            inputs = concatenate([skips[self.d - 1 - i], inputs], axis = 3)
            inputs = self.conv_up[self.d - 1 - i](inputs, training)
        
        inputs = self.map(inputs)
        
        if self.sig:
            inputs = sigmoid(inputs)
        model = Model(ins, inputs)
        plot_model(model)
        print(model.summary())