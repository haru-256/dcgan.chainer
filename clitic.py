import chainer
from chainer import Variable
import chainer.functions as F
import chainer.links as L


class Clitic(chainer.Chain):
    """Clitic

    build Clitic model

    Parametors
    ---------------------

    """

    def __init__(self, in_ch=1):
        super(Clitic, self).__init__()
        with self.init_scope():
            # initializers
            linear_init = chainer.initializers.Normal(0.0, 0.02)
            conv_init = chainer.initializers.No
