from keras.models import Sequential, Model
from keras.layers import Reshape, Conv2D, LeakyReLU, BatchNormalization, Activation, Input, Lambda, Concatenate, Layer
import keras.backend as k


class DiscriminatorMgr:
    __model = None

    @staticmethod
    def get_model():
        if DiscriminatorMgr.__model is not None:
            return DiscriminatorMgr.__model

        img_based_model = Sequential()

        # C64-C128-C256-C512

        # 3 for image and 1 for condition edge
        img_based_model.add(Conv2D(filters=64, kernel_size=4, strides=2, padding='VALID', input_shape=(70, 70, 6)))
        img_based_model.add(LeakyReLU(0.2))

        img_based_model.add(Conv2D(filters=128, kernel_size=4, strides=2, padding='VALID'))
        img_based_model.add(BatchNormalization())
        img_based_model.add(LeakyReLU(0.2))

        img_based_model.add(Conv2D(filters=256, kernel_size=4, strides=2, padding='VALID'))
        img_based_model.add(BatchNormalization())
        img_based_model.add(LeakyReLU(0.2))

        img_based_model.add(Conv2D(filters=512, kernel_size=4, strides=1, padding='VALID'))
        img_based_model.add(BatchNormalization())
        img_based_model.add(LeakyReLU(0.2))

        img_based_model.add(Conv2D(filters=1, kernel_size=4, strides=1, padding='VALID'))
        img_based_model.add(Activation('sigmoid'))

        DiscriminatorMgr.__model = img_based_model
        # DiscriminatorMgr.__model = img_based_model

        return img_based_model


if __name__ == '__main__':
    d = DiscriminatorMgr.get_model()
    d.summary()
