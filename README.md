# StyleGAN

Minimal implementation of StyleGAN in PyTorch.


Ref:

1. Karras, Laine, Aila - A style-based generator architecture for generative adversarial networks. [[CVPR 2019]](http://openaccess.thecvf.com/content_CVPR_2019/html/Karras_A_Style-Based_Generator_Architecture_for_Generative_Adversarial_Networks_CVPR_2019_paper.html) [[arXiv 2018]](https://arxiv.org/abs/1812.04948)


To implement:
- [x] Minibatch SD layer in discriminator
- [ ] 10% reduced learning rate for mapping_net
- [ ] Check which loss
- [ ] Progressive growing
- [ ] Style mixing regularization
- [ ] Truncation trick regularization
- [ ] Certain very specific parameter initializations
- [ ] Sampling demo notebook: `notebooks/Sampling.ipynb`
- [ ] Inversion demo notebook: `notebooks/Inversion.ipynb`