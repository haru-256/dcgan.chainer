import chainer
from chainer import training
from chainer.training import extensions
from generator import Generator
from visualize import out_generated_image
import argparse
import matplotlib.pyplot as plt
plt.style.use("ggplot")


if __name__ == '__main__':
    # パーサーを作る
    parser = argparse.ArgumentParser(
        prog='argparseTest',  # プログラム名
        usage='Demonstration of argparser',  # プログラムの利用方法
        description='description',  # 引数のヘルプの前に表示
        epilog='end',  # 引数のヘルプの後で表示
        add_help=True,  # -h/–help オプションの追加
    )

    # 引数の追加
    parser.add_argument('-s', '--seed', help='seed',
                        type=int, required=True)
    parser.add_argument('-n', '--number', help='the number of experiments.',
                        type=int, required=True)
    parser.add_argument('--hidden', help='the number of codes of Generator.',
                        type=int, default=100)
    parser.add_argument('-e', '--epoch', help='the number of epoch, defalut value is 100',
                        type=int, default=100)
    parser.add_argument('-bs', '--batch_size', help='batch size. defalut value is 128',
                        type=int, default=120)
    parser.add_argument('-g', '--gpu', help='specify gpu by this number. defalut value is 0',
                        choices=[0, 1], type=int, default=0)
    parser.add_argument('-dis', '--discriminator',
                        help='specify discriminator by this number. any of following;'
                        ' 0: original, 1: minibatch discriminatio, 2: feature matching. defalut value is 0',
                        choices=[0, 1, 2], type=int, default=0)
    parser.add_argument('-V', '--version', version='%(prog)s 1.0.0',
                        action='version',
                        default=False)

    # 引数を解析する
    args = parser.parse_args()

    gpu = args.gpu
    batch_size = args.batch_size
    n_hidden = args.hidden
    epoch = args.epoch
    seed = args.seed
    number = args.number  # number of experiments
    out = "result_{0}/result_{0}_{1}".format(number, seed)

    print('GPU: {}'.format(gpu))
    print('# Minibatch-size: {}'.format(batch_size))
    print('# n_hidden: {}'.format(n_hidden))
    print('# epoch: {}'.format(epoch))
    print('# out: {}'.format(out))

    # import discrimination
    if args.discriminator == 0:
        print("# Original Discriminator")
        from discriminator import Discriminator
        from updater import DCGANUpdater
    elif args.discriminator == 1:
        print("# Discriminator applied Minibatch Discrimination")
        from discriminator_md import Discriminator
        from updater import DCGANUpdater
    elif args.discriminator == 2:
        print("# Discriminator applied matching")
        from discriminator_fm import Discriminator
        from updater_fm import DCGANUpdatera

    print('')

    # Set up a neural network to train
    gen = Generator(n_hidden=n_hidden)
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
    trainer.extend(extensions.dump_graph("gen/loss", out_name="gen.dot"))
    trainer.extend(extensions.dump_graph("dis/loss", out_name="dis.dot"))
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
            file_name='loss_{0}_{1}.jpg'.format(number, seed),
            grid=False))  # grid=Falseとしているのは"ggplot"ではすでにgridが書かれているため．
    # 詳細は https://github.com/chainer/chainer/blob/v4.1.0/chainer/training/extensions/plot_report.py#L153-L154 参照

    # Run the training
    trainer.run()
