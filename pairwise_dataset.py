import chainer


class PairwiseDataset(chainer.dataset.DatasetMixin):
    def __init__(self, blur_image_list, sharp_image_list, root='.'):
        self.blur_images = chainer.datasets.ImageDataset(
            paths=blur_image_list, root=root)
        self.sharp_images = chainer.datasets.ImageDataset(
            paths=sharp_image_list, root=root)

        assert len(self.blur_images) == len(self.sharp_images)

    def __len__(self):
        return len(self.blur_images)

    def get_example(self, i):
        return self.blur_images[i] / 255, self.sharp_images[i] / 255
