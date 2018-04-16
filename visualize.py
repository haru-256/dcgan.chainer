#!/usr/bin/env python

import os

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import chainer
import chainer.backends.cuda
from chainer import Variable


def combine_images(generated_images):
    total = generated_images.shape[0]
    cols = int(np.sqrt(total))
    rows = int(np.ceil(float(total) / cols))
    width, height = generated_images.shape[1:3]
    combined_image = np.zeros(
        (height * rows, width * cols), dtype=generated_images.dtype)

    for index, image in enumerate(generated_images):
        i = int(index / cols)
        j = index % cols
        combined_image[width*i:width*(i+1), height*j:height*(j+1)] =\
            image[:, :, 0]
    return combined_image


def out_generated_image(gen, dis, rows, cols, seed, dst):
    @chainer.training.make_extension()
    def make_image(trainer):
        np.random.seed(seed)
        n_images = rows * cols
        xp = gen.xp
        z = Variable(xp.asarray(gen.make_hidden(n_images)))
        with chainer.using_config('train', False):
            x = gen(z)
        x = chainer.backends.cuda.to_cpu(x.data)
        np.random.seed()

        x = x * 127.5 + 127.5
        x = x.transpose(0, 2, 3, 1) # NCHW->NHWCに変形
        x = combine_images(x)
        plt.imshow(x, cmap=plt.cm.gray)
        plt.axis("off")
        preview_dir = '{}/preview'.format(dst)
        preview_path = preview_dir +\
                       '/image_{:}epoch.png'.format(trainer.updater.epoch)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        plt.tight_layout()
        plt.savefig(preview_path)

    return make_image
