import chainer
from chainer import training
from chainer.training import extensions

from discriminator import Discriminator
from generator import Generator
# from updater import DCGANUpdater
from updater_origin import DCGANUpdater
from visualize import out_generated_image
import matplotlib.pyplot as plt
plt.style.use("ggplot")


def main():
    gpu = 0
    batch_size = 128
    n_hidden = 100
    epoch = 100
    seed = 0
    number = 1  # number of experiments
    out = "result_{0}_{1}".format(number, seed)

    print('GPU: {}'.format(gpu))
    print('# Minibatch-size: {}'.format(batch_size))
    print('# n_hidden: {}'.format(n_hidden))
    print('# epoch: {}'.format(epoch))
    print('# out: {}'.format(out))
    print('')

    # Set up a neural network to train
    gen = Generator()
    dis = Discriminator()

    if gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(gpu).use()
        gen.to_gpu()  # Copy the model to the GPU
        dis.to_gpu()

    # Setup an optimizer
    def make_optimizer(model, alpha=0.0002, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        # optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001), 'hook_dec')
        return optimizer

    opt_gen = make_optimizer(gen)
    opt_dis = make_optimizer(dis)

    # Load the mnist dataset
    train, _ = chainer.datasets.get_mnist(withlabel=False, scale=255., ndim=3)

    train_iter = chainer.iterators.SerialIterator(train, batch_size)

    # Set up a trainer
    updater = DCGANUpdater(
        models=(gen, dis),
        iterator=train_iter,
        optimizer={
            'gen': opt_gen,
            'dis': opt_dis
        },
        device=gpu)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out=out)

    snapshot_interval = (5, 'epoch')
    display_interval = (1, 'epoch')
    trainer.extend(
        extensions.snapshot(
            filename='snapshot_epoch_{.updater.epoch}.npz'),
        trigger=snapshot_interval)
    trainer.extend(
        extensions.snapshot_object(gen, 'gen_epoch_{.updater.epoch}.npz'),
        trigger=snapshot_interval)
    trainer.extend(
        extensions.snapshot_object(dis, 'dis_epoch_{.updater.epoch}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.LogReport())
    trainer.extend(
        extensions.PrintReport([
            'epoch',
            'iteration',
            'gen/loss',
            'dis/loss',
        ]),
        trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=50))
    trainer.extend(
        out_generated_image(gen, dis, 7, 7, seed, out),
        trigger=display_interval)
    trainer.extend(
        extensions.PlotReport(
            ['gen/loss', 'dis/loss'],
            x_key='epoch',
            file_name='loss.jpg',
            grid=False))  # grid=Falseとしているのは"ggplot"ではすでにgridが書かれているため．
    # 詳細は https://github.com/chainer/chainer/blob/v4.1.0/chainer/training/extensions/plot_report.py#L153-L154 参照

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
