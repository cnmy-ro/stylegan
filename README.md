# StyleGAN

Minimal implementation of StyleGAN in PyTorch.


Ref:

1. Karras, Laine, Aila - A style-based generator architecture for generative adversarial networks. [[CVPR 2019]](http://openaccess.thecvf.com/content_CVPR_2019/html/Karras_A_Style-Based_Generator_Architecture_for_Generative_Adversarial_Networks_CVPR_2019_paper.html) [[arXiv 2018]](https://arxiv.org/abs/1812.04948)


To implement:
- [x] Minibatch SD layer in discriminator
- [ ] 1% reduced learning rate for mapping_net compared to synthesis_net
- [ ] Equalized learning rate
- [ ] Check which loss
- [ ] Progressive growing
- [ ] Style mixing regularization
- [ ] Truncation trick regularization
- [x] Certain very specific parameter initializations:
    - We initialize all weights of the [x] convolutional, [x] fully-connected, and [x] affine transform layers using N(0; 1).
    - The [x] constant input in synthesis network is initialized to one.
    - The [x] biases and [x] noise scaling factors are initialized to zero,
    - except for the [x] biases associated with ys that we initialize to one.
- [ ] Sampling demo notebook: `notebooks/Sampling.ipynb`
- [ ] Inversion demo notebook: `notebooks/Inversion.ipynb`