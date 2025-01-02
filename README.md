# Tree-Structured Shading Decomposition

### [Project Page](https://chen-geng.com/inv-shade-trees) | [Video](https://www.youtube.com/watch?v=L7zD9zM_zcg) | [Paper](https://chen-geng.com/files/inv-shade-trees.pdf) 

![ist](ist.gif)

> [Tree-Structured Shading Decomposition](https://chen-geng.com/inv-shade-trees)
>
> Chen Geng\*, Hong-Xing Yu*, Sharon Zhang, Maneesh Agrawala, Jiajun Wu (* denotes equal contribution)
>
> Stanford University
>
> ICCV 2023

Official implementation for the paper "Tree-Structured Shading Decomposition", which proposes a method to decompose a tree-structured representation for object shadings.

## Installation

Follow the steps below to set up the environment:

```bash
conda create -n InvShadeTrees python=3.9
conda activate InvShadeTrees
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install python-graphviz
pip install -r requirements.txt
```

Next, download the model checkpoints from [this link](https://drive.google.com/file/d/1aMGy7Cc3-CVjikmG3Px6rMzyQ6HwuUQw/view?usp=sharing) and extract them into the `ckpts` directory. The folder structure should look like this:

```bash
ckpts:
  bottom_network.pth
  classify_network.pth
  opparam_network.pth
  ops_network.pth
  sibling_bottom_network.pth
  sibling_top_network.pth
  top_network.pth
  vqvae_network.pth
```


## Inference

To perform inference, use the following command:

```bash
python run.py --cfg_file configs/topdown.yaml exp_name topdown gpus 0, demo_path examples/04.png vis_name ex04
```

Replace `examples/00.png` with the path to the shading image you want to decompose.

The results will be saved at `exps/ist/topdown/ex04/tree/graph/000000.png`. Note that the model's output is stochastic, so running the script multiple times may produce slightly different decomposition results.

Ensure that your shading map is formatted like those in the `examples/` folder. Specifically, the shading must be sphere-parameterized, with a mask identical to `mask.png`.

You can optionally fine-tune the decomposition result by performing an optimization with the following command:

```bash
python optim.py --cfg_file configs/optim.yaml exp_name ex04 mode inter result exps/ist/topdown/ex04/tree
python optim.py --cfg_file configs/optim.yaml exp_name ex04 mode leaf result exps/ist/topdown/ex04/tree
python optim.py --cfg_file configs/optim.yaml exp_name ex04 mode other_leaf result exps/ist/topdown/ex04/tree
python optim.py --cfg_file configs/optim.yaml exp_name ex04 mode bt result exps/ist/topdown/ex04/tree bt True
```

## Acknowledgements

This work was in part supported by Ford, NSF RI #2211258, AFOSR YIP FA9550-23-1-0127, the Toyota Research Institute (TRI), the Stanford Institute for Human-Centered AI (HAI), Amazon, and the Brown Institute for Media Innovation. 

The codebase builds upon ideas and implementations from the following projects:
- [instant-nvr](https://github.com/zju3dv/instant-nvr)
- [vq-vae-2-pytorch](https://github.com/rosinality/vq-vae-2-pytorch)
- [pytorch-classification](https://github.com/bearpaw/pytorch-classification)
- [pix2latent](https://github.com/minyoungg/pix2latent)

If you have any questions, feel free to contact us at **gengchen@cs.stanford.edu**.

## Citation

If you find our paper or code useful in your research, please consider citing us:

```
@inproceedings{geng2023shadetree,
  title={Tree-Structured Shading Decomposition},
  author={Chen Geng and Hong-Xing Yu and Sharon Zhang and Maneesh Agrawala and Jiajun Wu},
  booktitle={ICCV},
  year={2023}
}
```
