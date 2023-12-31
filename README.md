# StyleGAN

Concise implementation of StyleGAN in PyTorch.


---
To implement:
- [ ] Resuming training from a checkpoint
- [ ] Replace nearest upsampling and avgpool downsampling with low-pass filtering + bilinear sampling
- [ ] Model validation step: FID
- [x] R1 loss penalty
- [ ] Style mixing regularization
- [ ] Truncation trick regularization
- [x] Sampling demo notebook: `notebooks/Sampling.ipynb`
- [x] Inversion demo notebook: `notebooks/Inversion.ipynb`


---
References:
1. Karras, Laine, Aila - A style-based generator architecture for generative adversarial networks. [[arXiv 2018]](https://arxiv.org/abs/1812.04948) [[CVPR 2019]](http://openaccess.thecvf.com/content_CVPR_2019/html/Karras_A_Style-Based_Generator_Architecture_for_Generative_Adversarial_Networks_CVPR_2019_paper.html)
2. Karras, Aila, Laine, Lehtinen - Progressive growing of gans for improved quality, stability, and variation. [[arXiv 2017]](https://arxiv.org/abs/1710.10196) [[ICLR 2018]](https://openreview.net/forum?id=Hk99zCeAb&)
3. Zhang - Making convolutional networks shift-invariant again. [[ICML 2019]](https://arxiv.org/1904.11486)
4. Mescheder, Geiger, Nowozin - Which training methods for GANs do actually converge?. [[ICML 2018]](https://proceedings.mlr.press/v80/mescheder18a) [[arXiv 2018]](https://arxiv.org/pdf/1801.04406v4.pdf)