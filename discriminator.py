import chainer
from chainer import Variable
import chainer.functions as F
import chainer.links as L


class Discriminator(chainer.Chain):
    def __init__(self, bottom_width=3, ch=1, wscale=0.02):
        super(Discriminator, self).__init__()
        with self.init_scope():
            # initializers
            w = chainer.initializers.Normal(wscale)

            # register layer with variable
            self.c0 = L.Convolution2D(
                in_channels=ch,
                out_channels=64,
                ksize=5,
                stride=2,
                pad=2,
                initialW=w)
            self.c1 = L.Convolution2D(
                in_channels=None,
                out_channels=32,
                ksize=3,
                stride=2,
                pad=1,
                initialW=w)
            self.c2 = L.Convolution2D(
                in_channels=None,
                out_channels=16,
                ksize=3,
                stride=1,
                pad=1,
                initialW=w)

            # self.l4 = L.Linear(in_size=bottom_width*bottom_width*ch, out_size=1, initialW=w)
            self.l4 = L.Linear(in_size=None, out_size=1, initialW=w)

            # self.bn0 = L.BatchNormalization(size=ch//8, use_gamma=False)
            self.bn1 = L.BatchNormalization(size=32, use_gamma=False)
            self.bn2 = L.BatchNormalization(size=16, use_gamma=False)
            self.bn3 = L.BatchNormalization(size=8, use_gamma=False)

    def __call__(self, x):
        h = F.leaky_relu(self.c0(x))
        h = F.leaky_relu(self.bn1(self.c1(h)))
        h = F.leaky_relu(self.bn2(self.c2(h)))
        y = self.l4(h)  # conv->linear では勝手にreshapeが適用される

        return y


if __name__ == "__main__":
    import chainer.computational_graph as c
    from chainer import Variable
    import numpy as np

    z = np.random.uniform(-1, 1, (1, 1, 28, 28)).astype("f")
    model = Discriminator()
    img = model(Variable(z))
    # print(img)
    g = c.build_computational_graph(img)
    with open('dis_graph.dot', 'w') as o:
        o.write(g.dump())
