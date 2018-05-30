import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L


class Generator(chainer.Chain):
    """Generator

    build Generator model

    Parametors
    ---------------------
    n_hidden: int
       dims of random vector z

    bottom_width: int
       Width when converting the output of the first layer
       to the 4-dimensional tensor

    in_ch: int
       Channel when converting the output of the first layer
       to the 4-dimensional tensor

    Attributes
    ---------------------
    """

    def __init__(self, n_hidden=100, bottom_width=7, ch=128, wscale=0.02):
        super(Generator, self).__init__()
        self.n_hidden = n_hidden
        self.ch = ch
        self.bottom_width = bottom_width

        with self.init_scope():
            w = chainer.initializers.Normal(wscale)  # initializers

            self.l0 = L.Linear(in_size=self.n_hidden, out_size=bottom_width *
                               bottom_width * ch, initialW=w, nobias=True)
            # self.l1 = L.Linear(None, bottom_width * bottom_width * ch, initialW=w)
            self.dc1 = L.Deconvolution2D(
                in_channels=None,
                out_channels=ch // 2,
                ksize=4,
                stride=2,
                pad=1,
                initialW=w,
                nobias=True)  # (, 14, 14)
            """
            self.dc2 = L.Deconvolution2D(
                in_channels=None,
                out_channels=ch // 4,
                ksize=4,
                stride=2,
                pad=1,
                initialW=w,
                nobias=True)  # (, 14, 14)
            """
            self.dc2 = L.Deconvolution2D(
                in_channels=None,
                out_channels=1,
                ksize=4,
                stride=2,
                pad=1,
                initialW=w)  # (1, 28, 28)
            self.bn0 = L.BatchNormalization(bottom_width * bottom_width * ch)
            self.bn1 = L.BatchNormalization(ch // 2)
            # self.bn2 = L.BatchNormalization(ch // 4)

    def make_hidden(self, batchsize):
        """
        Function that makes z random vector in accordance with the uniform(-1, 1)

        batchsize: int
           batchsize indicate len(z)
        """
        return np.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1))\
                        .astype(np.float32)

    def __call__(self, z):
        h = F.relu(self.bn0(self.l0(z)))
        h = F.reshape(h, (len(z), self.ch, self.bottom_width,
                          self.bottom_width))  # dataformat is NCHW
        h = F.relu(self.bn1(self.dc1(h)))
        x = F.tanh(self.dc2(h))
        # h = F.relu(self.bn2(self.dc2(h)))
        # x = F.tanh(self.dc3(h))
        return x


if __name__ == "__main__":
    import chainer.computational_graph as c
    from chainer import Variable

    z = np.random.uniform(-1, 1, (1, 100)).astype("f")
    model = Generator()
    img = model(Variable(z))
    # print(img)
    g = c.build_computational_graph(img)
    with open('gen_graph.dot', 'w') as o:
        o.write(g.dump())
