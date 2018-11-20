from PIL import Image
import os
import numpy as np

if __name__ == '__main__':
    base_dir = 'D:/projects/data/pokemon/'
    a_dir = base_dir + 'pokemon-a/'
    b_dir = base_dir + 'pokemon-b/'
    x_dir = base_dir + 'pokemon-x/'
    y_dir = base_dir + 'pokemon-y/'
    print(os.listdir(a_dir))

    for filenames, dir_name in (os.listdir(a_dir), 'a'), (os.listdir(b_dir), 'b'):
        for filename in filenames:
            print(filename)
            img = Image.open(a_dir + filename)
            assert isinstance(img, Image.Image)
            img = img.convert('RGBA')

            img_ary = np.array(img)

            x, y = img_ary.shape[0:2]
            for i in range(x):
                for j in range(y):
                    r = img_ary[i, j, 0]
                    g = img_ary[i, j, 1]
                    b = img_ary[i, j, 2]
                    a = img_ary[i, j, 3]
                    if a == 0:
                        img_ary[i, j, :] = 255

            img = Image.fromarray(img_ary[:, :, 0:3], mode="RGB")
            img = img.resize(size=[256, 256])
            # img.show()
            img.save(y_dir + dir_name + filename + '.jpg')
            for i in range(x):
                for j in range(y):
                    r = img_ary[i, j, 0]
                    g = img_ary[i, j, 1]
                    b = img_ary[i, j, 2]
                    a = img_ary[i, j, 3]
                    if not (r < 30 and g < 30 and b < 30):
                        img_ary[i, j, 0] = 255
                    else:
                        img_ary[i, j, 0] = 0
            img_ary = img_ary[:, :, 0:1]
            img_ary = np.concatenate([img_ary, img_ary, img_ary], axis=-1)
            img = Image.fromarray(img_ary, mode="RGB")
            img = img.resize(size=(256, 256))
            img.save(x_dir + dir_name + filename + '.jpg')
            # img.show()
