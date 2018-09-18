# SRDenseNet (Caffe)
This is the implementation of paper: "T. Tong, G. Li, X. Liu, et al., 2017. Image super-resolution using dense skip connections. ICCV, p.4809-4817."
(http://openaccess.thecvf.com/content_ICCV_2017/papers/Tong_Image_Super-Resolution_Using_ICCV_2017_paper.pdf)

## Train
```
usage: main.py [-h] [--batchSize BATCHSIZE] [--nEpochs NEPOCHS] [--lr LR]
               [--step STEP] [--cuda] [--resume RESUME]
               [--start-epoch START_EPOCH] [--threads THREADS]
               [--pretrained PRETRAINED]

Pytorch SRDenseNet train

optional arguments:
  -h, --help            show this help message and exit
  --batchSize BATCHSIZE
                        training batch size
  --nEpochs NEPOCHS     number of epochs to train for
  --lr LR               Learning Rate. Default=1e-4
  --step STEP           Sets the learning rate to the initial LR decayed by
                        10 every n epochs, Default: n=30
  --cuda                Use cuda?
  --resume RESUME       Path to checkpoint (default: none)
  --start-epoch START_EPOCH
                        Manual epoch number (useful on restarts)
  --threads THREADS     Number of threads for data loader to use, Default: 1
  --pretrained PRETRAINED
                        path to pretrained model (default: none)

```
## Test
```
usage: test.py [-h] [--cuda] [--model MODEL] [--imageset IMAGESET] [--scale SCALE]

Pytorch SRDenseNet Test

optional arguments:
  -h, --help     show this help message and exit
  --cuda         use cuda?
  --model MODEL  model path
  --imageset IMAGESET  imageset name
  --scale SCALE  scale factor, Default: 4
```

### Prepare Training dataset
 The training data is generated with Matlab Bicubic Interplotation, please refer [Code for Data Generation](https://github.com/wxywhu/SRDenseNet-pytorch/tree/master/data) for creating training files.

### Prepare Test dataset
 The test imageset is generated with Matlab Bicubic Interplotation, please refer [Code for test](https://github.com/wxywhu/SRDenseNet-pytorch/tree/master/TestSet) for creating test imageset.
 
### Performance
 We provide a pretrained .[SRDenseNet x4 model](https://pan.baidu.com/s/1kkuS4sEDe-KyLBKpkKzXXg) trained on DIV2K images from [DIV2K_train_HR] (http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip).While I use the SR_DenseNet to train this model, so the performance is test based on this code.
 
 Non-overlapping sub-images with a size of 96 ¡Á 96 were cropped in the HR space.
 Other settings is the same as the original paper
 
 - Performance in terms of PSNR/SSIM on datasets Set5, Set14, BSD100, and Urban100
  
| DataSet/Method  | Bicubic interpolation | SRDenseNet |
| --------- |:-------------:|:----------------:|
| Set5      | 28.42/0.8103  | **30.44/0.8620** |
| Set14     | 26.00/0.7018  | **27.48/0.7518** |
| BSDS100   | 25.96/0.6674  | **26.91/0.7120** |
| Urban100	| 23.14/0.6570	| **24.43/0.7194** |

Our results are not as good as those presented in paper. Our code needs further improvement.

If you have any suggestion or question, please do not hesitate to contact me.

## Contact 
Ph.D. candidate, Shengke Xue
College of Information Science and Electronic Engineering
Zhejiang University, Hangzhou, P.R. China
Email: xueshengke@zju.edu.cn; xueshengke1993@gmail.com