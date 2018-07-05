import chainer
import chainercv


class Transform():
    def __call__(self, in_data):
        blur, sharp = in_data
        # flip, cropあたりのaugmentationはやっても問題なさそう
        # perspective transformみたいなピクセル値の補完が入る処理は一旦やめとく
        blur, param = chainercv.transforms.random_crop(
            blur, (512, 512), return_param=True, copy=True)
        sharp = sharp[:, param['y_slice'], param['x_slice']].copy()

        blur, param = chainercv.transforms.random_flip(
            blur, y_random=True, x_random=True, return_param=True)
        sharp = chainercv.transforms.flip(sharp, **param)

        return blur, sharp
