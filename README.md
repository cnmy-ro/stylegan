# StyleGAN

Concise implementation of StyleGAN and ProgressiveGAN in PyTorch.

Code organization:
- `stylegan`: neural net classes, dataloaders, training script, config file
- `notebooks`: sampling and inversion demo using a trained StyleGAN model

Training:
1. Configure the training parameters in `./stylegan/config.py`
2. Run:
    ```
    python ./stylegan/train.py
    ```

References:
1. Karras, Laine, Aila - A style-based generator architecture for generative adversarial networks. [[arXiv 2018]](https://arxiv.org/abs/1812.04948) [[CVPR 2019]](http://openaccess.thecvf.com/content_CVPR_2019/html/Karras_A_Style-Based_Generator_Architecture_for_Generative_Adversarial_Networks_CVPR_2019_paper.html)
2. Karras, Aila, Laine, Lehtinen - Progressive growing of gans for improved quality, stability, and variation. [[arXiv 2017]](https://arxiv.org/abs/1710.10196) [[ICLR 2018]](https://openreview.net/forum?id=Hk99zCeAb&)
3. Zhang - Making convolutional networks shift-invariant again. [[ICML 2019]](https://arxiv.org/1904.11486)
4. Mescheder, Geiger, Nowozin - Which training methods for GANs do actually converge?. [[ICML 2018]](https://proceedings.mlr.press/v80/mescheder18a) [[arXiv 2018]](https://arxiv.org/pdf/1801.04406v4.pdf)