# Active Bias - Training More Accurate Neural Networks by Emphasizing High Variance Samples
Unofficial tensorflow implementation of [*Active Bias*](http://papers.nips.cc/paper/6701-active-bias-training-more-accurate-neural-networks-by-emphasizing-high-variance-samples). Specifically, we implemented *SGD-WPV (SGD Weighted by Prediction Variance)*, which is one of the methods using the concept of high variance in the paper. In this repository, we call it *Active Bias* for convenience.

> __Publication__ </br>
> Chang, H. S., Learned-Miller, E., & McCallum, A., Active Bias: Training More Accurate Neural Networks by Emphasizing High Variance Samples," *Advances in Neural Information Processing Systems (NIPS)*, pp.
1002–1012, 2017.

## 1. Summary
For more accurate training, *Active Bias* emphasizes uncertain samples with high prediction variances. As shown in below figure borrowed from the [*original paper*](http://papers.nips.cc/paper/6701-active-bias-training-more-accurate-neural-networks-by-emphasizing-high-variance-samples), samples with low variances are too easy or too hard for training, therefore *Active Bias* focuses on uncertain samples with high variance. That is, the sample having highly variate prediction history is preferred to construct next mini-batch. In detail, *Active Bias* maintains a history structure to store all softmax outputs (or probabilities) of true labels, and computes the variance of accumulated prababilities for each sample. Then, based on the variance, it reweights the backward loss of the samples in each mini-batch before backward propagation.

<p align="center">
<img src="figures/overview.png " width="650">
</p>

## 2. Experimental Setup and Network Architecture
- To validate the superiority of *Active Bias*, we showed the test error on a simulated noisy CIFAR-10, which has mislabeled samples.
- To inject noisy labels, the true label *i* was flipped to the randomly chosen label *j* with a probability *tau*. That is, *tau* is a given noise rate that determines the degree of noiseness on dataset.
- A densely connected neural networks (L=40, k=12)([Huang et al./ 2017](http://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.html)) was used to train the noisy CIFAR-10.
- For the performance comparison, we compared the test loss of *Active Bias* with that of *Default*. *Defualt* trained the noisy CIFAR-10 without any processing for noisy labels.

## 3. Environment
- Python 3.6.4
- Tensorflow-gpu 1.8.0 (pip install tensorflow-gpu==1.8.0)
- Tensorpack (pip install tensorpack)
