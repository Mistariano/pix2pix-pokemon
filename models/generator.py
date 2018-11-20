from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, ReLU, BatchNormalization, Concatenate, Dropout, \
    Activation


class GeneratorMgr:
    __model = None

    @staticmethod
    def get_model():
        if GeneratorMgr.__model is not None:
            return GeneratorMgr.__model

        def bnorm_lrelu(x):
            return LeakyReLU(0.2)(BatchNormalization()(x))

        # ========== Encoder ==========

        # 3 for noise vector z and 1 for condition x
        input_layer = Input(shape=(256, 256, 6))

        # e1 256
        e1 = Conv2D(filters=64, kernel_size=4, strides=2, padding='SAME')(input_layer)
        e1_act = LeakyReLU(0.2)(e1)

        # e2 128
        e2 = Conv2D(filters=128, kernel_size=4, strides=2, padding='SAME')(e1_act)
        e2_act = bnorm_lrelu(e2)

        # e3 64
        e3 = (Conv2D(filters=256, kernel_size=4, strides=2, padding='SAME')(e2_act))
        e3_act = bnorm_lrelu(e3)

        # e4 32
        e4 = Conv2D(filters=512, kernel_size=4, strides=2, padding='SAME')(e3_act)
        e4_act = bnorm_lrelu(e4)

        # e5 16
        e5 = Conv2D(filters=512, kernel_size=4, strides=2, padding='SAME')(e4_act)
        e5_act = bnorm_lrelu(e5)

        # e6 8
        e6 = Conv2D(filters=512, kernel_size=4, strides=2, padding='SAME')(e5_act)
        e6_act = bnorm_lrelu(e6)

        # e7 4
        e7 = Conv2D(filters=512, kernel_size=4, strides=2, padding='SAME')(e6_act)
        e7_act = bnorm_lrelu(e7)

        # e8 2
        e8 = Conv2D(filters=512, kernel_size=4, strides=2, padding='SAME')(e7_act)
        e8_act = ReLU()(e8)

        # ========== Decoder ==========

        # d1
        d1 = Conv2DTranspose(filters=512, kernel_size=4, strides=2, padding='SAME')(e8_act)
        d1 = BatchNormalization()(d1)
        d1 = Dropout(0.5)(d1)
        d1 = Concatenate()([d1, e7])
        d1 = ReLU()(d1)

        # d2
        d2 = Conv2DTranspose(filters=512, kernel_size=4, strides=2, padding='SAME')(d1)
        d2 = BatchNormalization()(d2)
        d2 = Dropout(0.5)(d2)
        d2 = Concatenate()([d2, e6])
        d2 = ReLU()(d2)

        # d3
        d3 = Conv2DTranspose(filters=512, kernel_size=4, strides=2, padding='SAME')(d2)
        d3 = BatchNormalization()(d3)
        d3 = Dropout(0.5)(d3)
        d3 = Concatenate()([d3, e5])
        d3 = ReLU()(d3)

        # d4
        d4 = Conv2DTranspose(filters=512, kernel_size=4, strides=2, padding='SAME')(d3)
        d4 = BatchNormalization()(d4)
        d4 = Concatenate()([d4, e4])
        d4 = ReLU()(d4)

        # d5
        d5 = Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding='SAME')(d4)
        d5 = BatchNormalization()(d5)
        d5 = Concatenate()([d5, e3])
        d5 = ReLU()(d5)

        # d6
        d6 = Conv2DTranspose(filters=128, kernel_size=4, strides=2, padding='SAME')(d5)
        d6 = BatchNormalization()(d6)
        d6 = Concatenate()([d6, e2])
        d6 = ReLU()(d6)

        # d7
        d7 = Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='SAME')(d6)
        d7 = BatchNormalization()(d7)
        d7 = Concatenate()([d7, e1])
        d7 = ReLU()(d7)

        # d8
        d8 = Conv2DTranspose(filters=3, kernel_size=4, strides=2, padding='SAME')(d7)

        # out_layer
        out_layer = Activation(activation='tanh')(d8)

        GeneratorMgr.__model = Model(inputs=input_layer, outputs=out_layer)
        return GeneratorMgr.__model


if __name__ == '__main__':
    g = GeneratorMgr.get_model()
    g.summary()
