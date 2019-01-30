# Active Bias - Training More Accurate Neural Networks by Emphasizing High Variance Samples
Unofficial tensorflow implementation of [*Active Bias*](http://papers.nips.cc/paper/6701-active-bias-training-more-accurate-neural-networks-by-emphasizing-high-variance-samples). Specifically, we implemented *SGD-WPV (SGD Weighted by Prediction Variance)*, which is one of the methods using the concept of high variance in the paper. In this repository, we call it *Active Bias* for convenience.

> __Publication__ </br>
> Chang, H. S., Learned-Miller, E., & McCallum, A., Active Bias: Training More Accurate Neural Networks by Emphasizing High Variance Samples," *Advances in Neural Information Processing Systems (NIPS)*, pp.
1002–1012, 2017.

## 1. Summary
For more accurate training, *Active Bias* emphasizes uncertain samples with high prediction variances. As shown in below figure borrowed from the [*original paper*](http://papers.nips.cc/paper/6701-active-bias-training-more-accurate-neural-networks-by-emphasizing-high-variance-samples), samples with low variances are too easy or too hard for training, therefore *Active Bias* focuses on uncertain samples with high variance. That is, the sample having highly variate prediction probability is preferred to construct next mini-batch.

<p align="center">
<img src="figures/overview.png " width="650">
</p>
