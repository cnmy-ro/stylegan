# StyleGAN

Concise implementation of StyleGAN in PyTorch.


---
To implement:
- [ ] Replace nearest upsampling and avgpool downsampling with low-pass filtering + bilinear sampling
- [ ] Model validation step: FID
- [x] R1 loss penalty
- [ ] Style mixing regularization
- [ ] Truncation trick regularization
- [ ] Sampling demo notebook: `notebooks/Sampling.ipynb`
- [ ] Inversion demo notebook: `notebooks/Inversion.ipynb`

Issues:
- [ ] Model mode-collapses. Solve this.


---
References:
1. Karras, Laine, Aila - A style-based generator architecture for generative adversarial networks. [[arXiv 2018]](https://arxiv.org/abs/1812.04948) [[CVPR 2019]](http://openaccess.thecvf.com/content_CVPR_2019/html/Karras_A_Style-Based_Generator_Architecture_for_Generative_Adversarial_Networks_CVPR_2019_paper.html)
2. Karras, Aila, Laine, Lehtinen - Progressive growing of gans for improved quality, stability, and variation. [[arXiv 2017]](https://arxiv.org/abs/1710.10196) [[ICLR 2018]](https://openreview.net/forum?id=Hk99zCeAb&)
3. Zhang - Making convolutional networks shift-invariant again. [[ICML 2019]](https://arxiv.org/1904.11486)