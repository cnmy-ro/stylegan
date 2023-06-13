# StyleGAN

Minimal implementation of StyleGAN in PyTorch.


Reference:
1. Karras, Laine, Aila - A style-based generator architecture for generative adversarial networks. [[CVPR 2019]](http://openaccess.thecvf.com/content_CVPR_2019/html/Karras_A_Style-Based_Generator_Architecture_for_Generative_Adversarial_Networks_CVPR_2019_paper.html) [[arXiv 2018]](https://arxiv.org/abs/1812.04948)


To implement:
- [x] Minibatch SD layer in discriminator
- [x] Equalized learning rate
- [x] 1% reduced learning rate for mapping_net compared to synthesis_net
- [ ] Check which loss
- [x] Progressive growing
- [ ] Style mixing regularization
- [ ] Truncation trick regularization
- [x] Parameter initialization
- [ ] Sampling demo notebook: `notebooks/Sampling.ipynb`
- [ ] Inversion demo notebook: `notebooks/Inversion.ipynb`