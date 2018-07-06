import argparse
import random
from pathlib import Path

import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer.training import extensions

from chainerui.utils import save_args
from chainerui.extensions import CommandsExtension

import pairwise_dataset
import transform
import srcnn


def set_random_seed(seed):
    """
    https://qiita.com/TokyoMickey/items/cc8cd43545f2656b1cbd
    """

    # set Python random seed
    random.seed(seed)

    # set NumPy random seed
    np.random.seed(seed)

    # set Chainer(CuPy) random seed
    if chainer.backends.cuda.available:
        chainer.backends.cuda.cupy.random.seed(seed)


def psnr(pred, truth):
    """
    https://qiita.com/yoya/items/510043d836c9f2f0fe2f
    """
    batch_size = len(pred)
    mse = F.sum((pred-truth).reshape(batch_size, -1)**2, axis=1)
    mse = F.clip(mse, 1e-8, 1e+8)
    max_i = F.max(truth.reshape(batch_size, -1), axis=1)
    max_i = F.clip(max_i, 1e-8, 1e+8)
    return 20 * F.log10(max_i) - 10 * F.log10(mse)


def main():
    parser = argparse.ArgumentParser(description='Train Deblur Network')
    parser.add_argument('--seed', '-s', type=int, default=0,
                        help='seed for random values')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--learnrate', '-l', type=float, default=0.1,
                        help='Learning rate for SGD')
    parser.add_argument('--epoch', '-e', type=int, default=50,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--grad-clip', type=float, default=0.1,
                        help='')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print(args)
    print('')

    set_random_seed(args.seed)

    predictor = srcnn.create_srcnn()
    model = L.Classifier(
        predictor, lossfun=F.mean_squared_error, accfun=psnr)

    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    optimizer = chainer.optimizers.MomentumSGD(args.learnrate)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))
    optimizer.add_hook(
        chainer.optimizer_hooks.GradientClipping(args.grad_clip))

    base_dir = 'data/blurred_sharp'
    train_data = pairwise_dataset.PairwiseDataset(blur_image_list=str(Path(base_dir).joinpath('train_blur_images.txt')),
                                                  sharp_image_list=str(
                                                      Path(base_dir).joinpath('train_sharp_images.txt')),
                                                  root=base_dir)
    train_data = chainer.datasets.TransformDataset(
        train_data, transform.Transform())

    test_data = pairwise_dataset.PairwiseDataset(blur_image_list=str(Path(base_dir).joinpath('test_blur_images.txt')),
                                                 sharp_image_list=str(
                                                     Path(base_dir).joinpath('test_sharp_images.txt')),
                                                 root=base_dir)
    # 普通はTransformしないような気がするけど、解像度がかわっちゃうのがなー
    test_data = chainer.datasets.TransformDataset(
        test_data, transform.Transform())

    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test_data, args.batchsize,
                                                 repeat=False, shuffle=False)
    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.FailOnNonNumber())
    # Evaluate the model with the test dataset for each epoch
    eval_trigger = (1, 'epoch')
    trainer.extend(extensions.Evaluator(test_iter, model,
                                        device=args.gpu), trigger=eval_trigger)

    # Reduce the learning rate by half every 25 epochs.
    lr_drop_epoch = [int(args.epoch*0.5), int(args.epoch*0.75)]
    lr_drop_ratio = 0.1
    print('lr schedule: {}, timing: {}'.format(lr_drop_ratio, lr_drop_epoch))

    def lr_drop(trainer):
        trainer.updater.get_optimizer('main').lr *= lr_drop_ratio
    trainer.extend(
        lr_drop,
        trigger=chainer.training.triggers.ManualScheduleTrigger(lr_drop_epoch, 'epoch'))
    trainer.extend(extensions.observe_lr(), trigger=(1, 'epoch'))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot at each epoch
    trainer.extend(extensions.snapshot_object(model.predictor,
                                              'model_{.updater.epoch}.npz'), trigger=(1, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(trigger=(100, 'iteration')))

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'lr', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']), trigger=(100, 'iteration'))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())
    # interact with chainerui
    trainer.extend(CommandsExtension(), trigger=(100, 'iteration'))
    # save args
    save_args(args, args.out)

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
