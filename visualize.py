import pathlib
import numpy as np
import matplotlib.pyplot as plt
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
    @chainer.training.make_extension(trigger=(1, 'epoch'))
    def make_image(trainer):
        n_images = rows * cols

        np.random.seed(seed)  # fix seed
        xp = gen.xp  # get module

        # test, evaluationの時は以下の２つを設定しなければならない
        # https://qiita.com/mitmul/items/1e35fba085eb07a92560
        # 'train'をFalseにすることで，train時とテスト時で挙動が異なるlayer(BN, Dropout)
        # を制御する
        with chainer.using_config('train', False):
            # 'enable_backprop'をFalseとすることで，無駄な計算グラフの構築を行わない
            # ようにしメモリの消費量を抑える.
            with chainer.using_config('enable_backprop', False):
                z = Variable(xp.asarray(gen.make_hidden(n_images)))
                x = gen(z)

        x = chainer.backends.cuda.to_cpu(x.data)
        np.random.seed()

        x = (x * 127.5 + 127.5) / 255  # 0~255に戻し0~1へ変形
        x = x.transpose(0, 2, 3, 1)  # NCHW->NHWCに変形
        x = combine_images(x)

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
        axes.imshow(x, cmap=plt.cm.gray)
        axes.axis("off")
        preview_dir = pathlib.Path('{}/preview'.format(dst))
        preview_path = preview_dir /\
            'image_{:}epoch.jpg'.format(trainer.updater.epoch)
        if not preview_dir.exists():
            preview_dir.mkdir()
        axes.set_title("epoch: {}".format(trainer.updater.epoch), fontsize=18)
        fig.tight_layout()
        fig.savefig(preview_path)
        plt.close(fig)  # あまりに図が多いとメモリが圧迫されるので閉じる

    return make_image
