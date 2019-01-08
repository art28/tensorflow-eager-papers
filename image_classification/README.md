# Image Classification
Image Classification models used in [ILSVRC](http://www.image-net.org/challenges/LSVRC/).

## Dataset
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) is used because [original ILSVRC data](http://www.image-net.org/download-images) is too heavy to be handled in normal computers.
- so, most of the models' sizes are intentionally reduced.
- but, I tried to maintain main chracateristics.
- to download, run `python download_cifar.py`

## Categories
 - [x] [Alexnet](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
 - [x] [VGGNet](https://arxiv.org/pdf/1409.1556.pdf)
 - [x] [GoogLenet(inception)](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf)
 - [x] [Resnet](https://arxiv.org/pdf/1512.03385.pdf)
