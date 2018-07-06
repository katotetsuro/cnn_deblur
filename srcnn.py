import chainer
import chainer.links as L
import chainer.functions as F
from chainercv.links.connection.conv_2d_bn_activ import Conv2DBNActiv


def create_srcnn():
    """
    waifu2xのSRCNNというモデルを参考に作ってみる
    """
    return chainer.Sequential(
        Conv2DBNActiv(in_channels=None, out_channels=32, ksize=3,
                      activ=lambda x: F.leaky_relu(x, 0.1), pad=1),
        Conv2DBNActiv(in_channels=None, out_channels=32, ksize=3,
                      activ=lambda x: F.leaky_relu(x, 0.1), pad=1),
        Conv2DBNActiv(in_channels=None, out_channels=64, ksize=3,
                      activ=lambda x: F.leaky_relu(x, 0.1), pad=1),
        Conv2DBNActiv(in_channels=None, out_channels=64, ksize=3,
                      activ=lambda x: F.leaky_relu(x, 0.1), pad=1),
        Conv2DBNActiv(in_channels=None, out_channels=128, ksize=3,
                      activ=lambda x: F.leaky_relu(x, 0.1), pad=1),
        Conv2DBNActiv(in_channels=None, out_channels=128, ksize=3,
                      activ=lambda x: F.leaky_relu(x, 0.1), pad=1),
        L.Convolution2D(in_channels=None, out_channels=3, ksize=3, pad=1)
    )
