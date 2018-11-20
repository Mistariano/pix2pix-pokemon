from models.discriminator import DiscriminatorMgr
from models.generator import GeneratorMgr
import keras.backend as k
from keras.layers import Concatenate, Input, Lambda, Reshape
from keras.models import Model, Sequential
from keras.optimizers import sgd
from keras.callbacks import TensorBoard
from keras.utils.vis_utils import plot_model
from keras.losses import binary_crossentropy
import numpy as np


class CGANModelMgr:
    __CGAN_model = None
    __D_model = None
    __compiled = False
    __tb = None

    @staticmethod
    def get_models():
        """
        get the instances of CGAN model and its discriminator
        :return: CGAN, D
        """
        if CGANModelMgr.__CGAN_model is not None:
            assert isinstance(CGANModelMgr.__CGAN_model, Model)
            assert isinstance(CGANModelMgr.__D_model, Sequential)
            return CGANModelMgr.__CGAN_model, CGANModelMgr.__D_model

        if CGANModelMgr.__D_model is None:
            d = DiscriminatorMgr.get_model()
        else:
            d = CGANModelMgr.__D_model

        input_zx = Input(shape=[256, 256, 6])

        slice_x = lambda t: t[:, :, :, -3:]  # x: 256 X 256 X 1
        # input_z = Lambda(slice_z)(input_zx)
        input_x = Lambda(slice_x)(input_zx)

        g_net = GeneratorMgr.get_model()
        g_output = g_net(input_zx)  # 256 X 256 X 3

        concat_gzx_x = Concatenate(axis=-1)([g_output, input_x])  # 256 X 256 X 6

        d_output = Lambda(CGANModelMgr.__get_d_output, arguments={'d': d})(concat_gzx_x)

        # d.trainable = False
        # d_output = d(concat)

        # cgan = Model(input_zx, [g_output, d_output])
        cgan = Model(input_zx, [g_output, d_output])

        CGANModelMgr.__CGAN_model = cgan
        CGANModelMgr.__D_model = d

        return cgan, d

    @staticmethod
    def train_g(zx, y):
        """

        :param zx:
        :param y:
        :return:
        """
        CGANModelMgr.compile()
        gan, d = CGANModelMgr.get_models()

        # tb = CGANModelMgr.__tb
        assert isinstance(d, Model)
        assert isinstance(gan, Model)
        d.trainable = False
        return gan.train_on_batch(x=zx, y=y)

    @staticmethod
    def train_d(img_x, is_real):
        """
        (img_x, is_real)
        :param batch_generator:
        :return:
        """
        CGANModelMgr.compile()
        _, d = CGANModelMgr.get_models()

        tb = CGANModelMgr.__tb
        assert isinstance(d, Model)
        d.trainable = True
        return d.train_on_batch(x=img_x, y=is_real)

    @staticmethod
    def __d_loss(is_real, dgz):
        return -k.mean(is_real * k.log(dgz) + (1. - is_real) * k.log(1 - dgz))

    @staticmethod
    def __g_loss(y, gz):
        # return -k.mean(k.log(1 - gz))
        return binary_crossentropy(y, gz)

    @staticmethod
    def compile():
        cgan, d = CGANModelMgr.get_models()
        if not CGANModelMgr.__compiled:
            d.trainable = True
            d.compile(optimizer='adam', loss='binary_crossentropy')
            # d.compile(optimizer='adam', loss=CGANModelMgr.__d_loss)
            d.trainable = False
            # cgan.compile(optimizer='adam', loss=['mae', 'binary_crossentropy'], loss_weights=[1e-2, 3])
            cgan.compile(optimizer='adam', loss=['mae', 'binary_crossentropy'], loss_weights=[1e2, 1])
            CGANModelMgr.__compiled = True
        if CGANModelMgr.__tb is None:
            CGANModelMgr.__tb = TensorBoard()

    @staticmethod
    def extract_patches(x):
        list_patches = []
        x = k.expand_dims(x, 1)
        for i in range(0, 256 // 70):
            for j in range(0, 256 // 70):
                # [batch, i*70:(i+1)*70, j*70:(j+1)*70, :]
                patch = Lambda(lambda t: t[:, :, i * 70:(i + 1) * 70, j * 70:(j + 1) * 70, :])(x)
                list_patches.append(patch)
        concat = Concatenate(axis=1)(list_patches)
        return [concat[:, i, :, :, :] for i in range(concat.shape[1])]

    @staticmethod
    def __get_d_output(x, d):
        d.trainable = False
        patch_out_list = [d(patch) for patch in CGANModelMgr.extract_patches(x)]
        patch_out_cat = Concatenate(-1)(patch_out_list)
        return k.mean(patch_out_cat, axis=-1, keepdims=True)

    @staticmethod
    def generate(zx):
        cgan_, _ = CGANModelMgr.get_models()
        assert isinstance(cgan_, Model)
        return cgan_.predict_on_batch(zx)


if __name__ == '__main__':
    cgan_, d_ = CGANModelMgr.get_models()
    CGANModelMgr.compile()
    cgan_.summary()
    d_.trainable = True
    d_.build((70, 70, 3))
    d_.summary()
    plot_model(cgan_, show_shapes=True)
