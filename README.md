# StyleGAN

Simple, yet fully-featured, implementation of StyleGAN in PyTorch.


---
To implement:
- [x] In `train.py`, do prog growth based on #images instead of #iters
- [ ] Checkpointing
- [ ] Check which loss
- [x] Parameter initialization
- [x] Equalized learning rate
- [x] 1% reduced learning rate for mapping_net compared to synthesis_net
- [x] Progressive growing
- [ ] R1 loss penalty
- [ ] Style mixing regularization
- [ ] Truncation trick regularization
- [ ] Sampling demo notebook: `notebooks/Sampling.ipynb`
- [ ] Inversion demo notebook: `notebooks/Inversion.ipynb`


---
References:
1. Karras, Laine, Aila - A style-based generator architecture for generative adversarial networks. [[arXiv 2018]](https://arxiv.org/abs/1812.04948) [[CVPR 2019]](http://openaccess.thecvf.com/content_CVPR_2019/html/Karras_A_Style-Based_Generator_Architecture_for_Generative_Adversarial_Networks_CVPR_2019_paper.html)
2. Karras, Aila, Laine, Lehtinen - Progressive growing of gans for improved quality, stability, and variation. [[arXiv 2017]](https://arxiv.org/abs/1710.10196) [[ICLR 2018]](https://openreview.net/forum?id=Hk99zCeAb&)