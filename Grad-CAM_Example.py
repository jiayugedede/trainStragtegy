import tensorflow as tf
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 8,8

from scipy.ndimage.interpolation import zoom
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
import numpy as np
import os

from Grad_CAMutils import  preprocess_image, show_imgwithheat
from gradcam import grad_cam,grad_cam_plus


class MultiSpectralAttentionLayer(layers.Layer):
    def __init__(self, name, reduction=16, freq_sel_method='top16', **kwargs):
        super(MultiSpectralAttentionLayer, self).__init__(name=name)
        self.reduction = int(reduction)
        self.freq_sel_method = freq_sel_method
        super(MultiSpectralAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        _, h, w, c = input_shape
        self.channel = int(c)
        self.reduction = self.reduction
        self.fc1 = layers.Dense(self.channel / self.reduction, use_bias=True, activation='relu')
        self.fc2 = layers.Dense(self.channel, use_bias=True, activation="sigmoid")
        self.reshapeTensor = tf.keras.layers.Reshape((1, 1, c))
        self.mapper_x, self.mapper_y = self.get_freq_indices(self.freq_sel_method)
        self.mapper_x = [temp_x * (h // 7) for temp_x in self.mapper_x] #快速傅里叶计算位置重定位。
        self.mapper_y = [temp_y * (w // 7) for temp_y in self.mapper_y]
        self.dynamic_weight = tf.Variable(self.get_dct_filter(h, w, self.mapper_x, self.mapper_y, channel=c))
        super().build(input_shape)

    def get_freq_indices(self, methods):
        assert methods in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                           'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                           'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
        num_freq = int(methods[3:])

        if 'top' in methods:
            all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2,
                                 2,
                                 6, 1]
            all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3,
                                 0,
                                 5, 3]
            mapper_x = all_top_indices_x[:num_freq]
            mapper_y = all_top_indices_y[:num_freq]

        elif 'low' in methods:
            all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1,
                                 2,
                                 3, 4]

            all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6,
                                 5,
                                 4, 3]

            mapper_x = all_low_indices_x[:num_freq]
            mapper_y = all_low_indices_y[:num_freq]

        elif 'bot' in methods:
            all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5,
                                 5,
                                 3, 6]
            all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5,
                                 3,
                                 3, 3]
            mapper_x = all_bot_indices_x[:num_freq]
            mapper_y = all_bot_indices_y[:num_freq]
        else:
            raise NotImplementedError
        return mapper_x, mapper_y

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        
        # h, w, self.mapper_x, self.mapper_y, channel=c
        dct_numpy_filter = np.zeros(shape=(tile_size_y, tile_size_x, channel), dtype=np.float32)
        c_part = channel // len(mapper_x)

        for i, (u_x, u_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_numpy_filter[t_y,t_x, i * c_part: (i + 1) * c_part] = self.build_filter(t_x, u_x, tile_size_x) * \
                                                                               self.build_filter(t_y, u_y, tile_size_y)
        return dct_numpy_filter

    def build_filter(self, pos, frequency, POS):
        result = np.cos(np.pi * frequency * (pos + 0.5) / POS) / np.sqrt(POS)
        if frequency == 0:
            return result
        else:
            return result * np.sqrt(2)

    def get_config(self):
        base_config = super(MultiSpectralAttentionLayer, self).get_config()
        config = {
            "reduction": self.reduction,
            "freq_sel_method": self.freq_sel_method,
        }
        return dict(list(base_config.items()) + list(config.items()))

    @tf.function
    def call(self, inputs, training=None):
        x = inputs * self.dynamic_weight
        result = tf.math.reduce_sum(x, axis=[1, 2], keepdims=True)
        x = self.fc1(result)
        x = self.fc2(x)
        x = self.reshapeTensor(x)
        return inputs * x


# @tf.keras.utils.register_keras_serializable()
class HardSigmoid(tf.keras.layers.Layer):
    def __init__(self, name="HardSigmoid", **kwargs):
        super(HardSigmoid, self).__init__(name=name)
        self.relu6 = tf.keras.layers.ReLU(max_value=6, name="ReLU6", **kwargs)
        super(HardSigmoid, self).__init__(**kwargs)

    def call(self, input):
        return self.relu6(input + 3.0) / 6.0

    def get_config(self):
        base_config = super(HardSigmoid, self).get_config()
        return dict(list(base_config.items()))


class SeBlock(tf.keras.layers.Layer):
    def __init__(self, reduction=16, l2=2e-4, **kwargs):
        super(SeBlock, self).__init__(**kwargs)
        self.reduction = reduction
        self.l2 = l2

    def build(self, input_shape):
        _, h, w, c = input_shape
        self.gap = tf.keras.layers.GlobalAveragePooling2D(name=f'AvgPool{h}x{w}')
        self.fc1 = tf.keras.layers.Dense(units=c//self.reduction, activation="relu", use_bias=False,
                                         kernel_regularizer=tf.keras.regularizers.l2(self.l2), name="Squeeze")
        self.fc2 = tf.keras.layers.Dense(units=c, activation=HardSigmoid(), use_bias=False,
                                         kernel_regularizer=tf.keras.regularizers.l2(self.l2), name="Excite")
        self.reshape = tf.keras.layers.Reshape((1, 1, c), name=f'Reshape_None_1_1_{c}')

        super().build(input_shape)

    def call(self, input):
        output = self.gap(input)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.reshape(output)
        return input * output

    def get_config(self):
        config = {"reduction":self.reduction, "l2":self.l2}
        base_config = super(SeBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class DoubleBlock(tf.keras.layers.Layer):
    def __init__(self, name, reduction=16, **kwargs):
        super(DoubleBlock, self).__init__(name=name)
        self.reduction = reduction
        super(DoubleBlock, self).__init__(**kwargs)

    def build(self, input_shape):
        _, h, w, c = input_shape
        self.gap = tf.keras.layers.GlobalAveragePooling2D(name=f'AvgPool{h}x{w}')
        self.fc1 = tf.keras.layers.Dense(units=c//self.reduction, activation="relu",use_bias=True, name="Squeeze")
        self.fc2 = tf.keras.layers.Dense(units=c, activation=HardSigmoid(), use_bias=True, name="Excite")
        self.reshape = tf.keras.layers.Reshape((1, 1, c), name=f'Reshape_None_1_1_{c}')

        self.gapD = tf.keras.layers.GlobalAveragePooling2D(name=f'AvgPool{h}x{w}D')
        self.fc1D = tf.keras.layers.Dense(units=c//self.reduction, activation="relu",use_bias=True, name="SqueezeD")
        self.fc2D = tf.keras.layers.Dense(units=c, activation=HardSigmoid(), use_bias=True, name="ExciteD")
        self.reshapeD = tf.keras.layers.Reshape((1, 1, c), name=f'Reshape_None_1_1_{c}D')
        super().build(input_shape)

    def call(self, input):
        output = self.gap(input)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.reshape(output)
        output = input * output
        output = self.gapD(output)
        output = self.fc1D(output)
        output = self.fc2D(output)
        output = self.reshapeD(output)
        return input * output

    def get_config(self):
        config = {"reduction": self.reduction}
        base_config = super(DoubleBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def load_trained_model():
    _custom_objects = {
        "Custom>SeBlock" : SeBlock,
        "Custom>MultiSpectralAttentionLayer": MultiSpectralAttentionLayer,
        # "Custom>DoubleBlock":DoubleBlock,
        }
    
    model_name = r"F:\博士论文\源文件\ResnetFunctionMA\model.63-0.2748-1.h5"
    # model_name = r"F:\博士论文\源文件\ResnetFunctionDoubleSE\model.51-0.3163-2.h5"
    
    function_model = load_model(model_name, custom_objects=_custom_objects)

    print('model load success')

    return function_model

model = load_trained_model()
model.summary()

# img_path = r"F:\cassavaImg\CMD_1234571117.jpg"
img_path = r"F:\cassavaImg\CBSD_784.jpg"
img = preprocess_image(img_path)

heatmap = grad_cam(model, img,
                    label_name = ['CBB', 'CBSD', 'CGM', "CMD", "Healthy"],
                    #category_id = 0,
                    )
show_imgwithheat(img_path, heatmap)

# heatmap_plus = grad_cam_plus(model, img,
#                              label_name = ['CBB', 'CBSD', 'CGM', "CMD", "Healthy"]
#                              )
# show_imgwithheat(img_path, heatmap_plus)