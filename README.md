# CMA-Former

Official implementation of **Cascaded Multi-Head Attention Transformer Framework for Direction of Arrival Estimation**.

## Installation

```bash
conda create -n cmaformer python=3.10
conda activate cmaformer
pip install -r requirements.txt
```

Install the PyTorch version that matches your CUDA environment if GPU acceleration is needed.

## Usage

Generate the synthetic DOA dataset:

```bash
python data.py
```

Train CMA-Former:

```bash
python train.py
```

Run ablation training:

```bash
python train_ablation.py
```

Run the main experiment scripts:

```bash
python exp_diff_models_snr_enhanced.py
python exp_diff_models_interval_enhanced.py
python exp_diff_models_traversal_enhanced.py
python exp_diff_models_snapshot.py
python exp_nonideal.py
python exp_eff.py
python exp_diff_models_real.py
```

Checkpoints and pretrained weights are not included in this repository. If needed, place them in the paths expected by the scripts, for example:

```text
checkpoints/depth_4_embed_dim_128/best_model.pt
exp_utils/weights/*.pt
exp_utils/weights/*.pth
```

Generated datasets, checkpoints, and pretrained weights are ignored by Git.

## Usage Statement

This code is released only for scientific research and learning purposes. Commercial use or other use is not authorized without permission from the authors.

## Citation

If this code is useful for your research, please cite:

```bibtex
@ARTICLE{11603315,
  author={Su, Yunye and Wang, Xianpeng and Wen, Linqiang and Guo, Yuehao and Wang, Han and Chen, Gaojie},
  journal={IEEE Transactions on Cognitive Communications and Networking}, 
  title={Cascaded Multi-Head Attention Transformer Framework for Direction of Arrival Estimation}, 
  year={2026},
  volume={12},
  number={},
  pages={10450-10465},
  keywords={Direction-of-arrival estimation;Arrays;Modeling;Signal to noise ratio;Algorithms;Covariance matrices;Convolutional neural networks;Multiple signal classification;Music;Training;Direction of arrival (DOA) estimation;self-attention;transformer;deep neural network;deep learning (DL)},
  doi={10.1109/TCCN.2026.3712135}}
```
