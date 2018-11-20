from keras.models import Model
from keras.optimizers import adam
from keras.callbacks import TensorBoard
from models.cgan import CGANModelMgr
from models.generator import GeneratorMgr
from models.discriminator import DiscriminatorMgr
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
import datetime
import os


def load_img(path):
    img = image.load_img(path=path, target_size=(256, 256))
    arr = image.img_to_array(img)
    return arr


def read_img():
    base_dir = 'D://projects/data/pokemon/'
    x_dir = base_dir + 'pokemon-x/'
    y_dir = base_dir + 'pokemon-y/'
    for filename in os.listdir(x_dir):
        if filename in os.listdir(y_dir):
            x = encode_img(np.array(image.load_img(x_dir + filename)))
            x = np.expand_dims(x, 0)
            y = encode_img(np.array(image.load_img(y_dir + filename)))
            y = np.expand_dims(y, 0)
            print('Yield', filename)
            yield x, y


def g_mini_batch_generator():
    img_gen = read_img()
    for index in range(936 // 4):
        zx_list = []
        x_list = []
        y_list = []
        for _ in range(4):
            z = encode_img(np.random.normal(size=(1, 256, 256, 3)) * 255.)
            x, y = img_gen.__next__()
            zx = np.concatenate([z, x], -1)
            zx_list.append(zx)
            x_list.append(x)
            y_list.append(y)
        zx = np.concatenate(zx_list, 0)
        x = np.concatenate(x_list, 0)
        y = np.concatenate(y_list, 0)
        yield zx, x, y, index


def encode_img(x):
    return (x - 128.) / 128.


def decode_img(x):
    return (x * 128.) + 128.


def extract_patches(arr):
    list_patches = []
    arr = np.expand_dims(arr, 1)
    for i in range(0, 256 // 70):
        for j in range(0, 256 // 70):
            # [batch, i*70:(i+1)*70, j*70:(j+1)*70, :]
            patch = arr[:, :, i * 70:(i + 1) * 70, j * 70:(j + 1) * 70, :]
            list_patches.append(patch)
    concat = np.concatenate(list_patches, 1)
    return [concat[:, i, :, :, :] for i in range(concat.shape[1])]


def train():
    # CGANModelMgr.train_d()
    sess = tf.Session()
    logdir = "log/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/'
    writer = tf.summary.FileWriter(logdir, sess.graph)
    d_loss_ph = tf.placeholder(dtype=tf.float32, shape=[], name='d_loss_ph')
    g_loss_ph = tf.placeholder(dtype=tf.float32, shape=[], name='g_loss_ph')
    x_ph = tf.placeholder(dtype=tf.float32, shape=[None, 256, 256, 3], name='x_ph')
    y_ph = tf.placeholder(dtype=tf.float32, shape=[None, 256, 256, 3], name='y_ph')
    gz_ph = tf.placeholder(dtype=tf.float32, shape=[None, 256, 256, 3], name='y_ph')
    tf.summary.scalar("D_Loss", d_loss_ph)
    tf.summary.scalar("G_Loss", g_loss_ph)
    tf.summary.image("x", decode_img(x_ph), 5)
    tf.summary.image("y", decode_img(y_ph), 5)
    tf.summary.image("gz", decode_img(gz_ph), 5)
    for epoch in range(30):
        print('========== Epoch {} =========='.format(epoch))
        for zx, x, y, index in g_mini_batch_generator():
            # Labels are ones because we want to cheat the D net
            g_loss = CGANModelMgr.train_g(zx, [y, np.ones(shape=(4, 1, 1, 1))])
            print(g_loss)

            gz = CGANModelMgr.generate(zx)[0]
            gz_x, y_x = np.concatenate([gz, x], -1), np.concatenate([y, x], -1)

            gz_patches, y_patches = extract_patches(gz_x), extract_patches(y_x)

            labels = np.concatenate(
                [np.zeros(shape=[4 * len(gz_patches), 1, 1, 1]), np.ones([4 * len(y_patches), 1, 1, 1])], 0)
            patches = np.concatenate(gz_patches + y_patches, 0)
            d_loss = CGANModelMgr.train_d(patches, labels)
            print(d_loss)

            merged = tf.summary.merge_all()
            summary = sess.run(merged, feed_dict={d_loss_ph: d_loss, g_loss_ph: g_loss[0], x_ph: x, y_ph: y, gz_ph: gz})
            writer.add_summary(summary, epoch * (936 // 4) + index)


if __name__ == '__main__':
    train()
