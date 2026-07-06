# CMA-Former

Official code release for **Cascaded Multi-Head Attention Transformer Framework for Direction of Arrival Estimation**.

CMA-Former is a Transformer-based grid DOA estimator. It tokenizes the sample covariance matrix by taking only the upper-triangular off-diagonal sensor-pair correlations. Each token contains four features: real part, imaginary part, `sin(phase)`, and `cos(phase)`. The model then uses cascaded multi-head self-attention blocks with progressive head counts `[4, 8, 16]` and predicts source confidence over the angular grid `[-60, 60]` with 1-degree resolution.

## Repository Layout

```text
.
|-- models.py                          # CMAFormer model definition
|-- data.py                            # Synthetic ULA data generation
|-- train.py                           # Main CMA-Former training entry
|-- train_ablation.py                  # Component ablation training entry
|-- exp_utils/                         # Baseline models, MUSIC, and loaders
|-- exp_*.py                           # Paper experiment scripts
|-- checkpoints/                       # Optional local CMA-Former checkpoints, ignored by Git
|-- exp_utils/weights/                 # Optional local baseline weights, ignored by Git
`-- exp/                               # Saved paper-result figures/tables
```

The old 3-channel model, duplicate data generators, half-precision training prototype, and temporary plotting scripts have been removed from this release tree.

## Environment

```bash
conda create -n cmaformer python=3.10
conda activate cmaformer
pip install -r requirements.txt
```

For GPU experiments, install the PyTorch build that matches your CUDA version from the official PyTorch instructions.

## Data

The paper uses a 16-element ULA, angular grid `[-60, 60]`, SNR values `[-20, -15, -10, -5, 0]` dB, and `1000` snapshots. The full balanced training set contains single-, dual-, and triple-target cases.

Generate the synthetic training set:

```bash
python data.py
```

By default, `data.py` writes:

```text
DATA/doa_signal_dataset_lite_hard.h5
```

The generated HDF5 files are intentionally ignored by Git. To run a smaller smoke test, reduce `DATA_RATIO` in `data.py`.

## Training

Train the main CMA-Former model:

```bash
python train.py
```

The default configuration follows the paper-style main model:

```text
depth = 4
embed_dim = 128
num_heads = [4, 8, 16]
batch_size = 64
epochs = 100
optimizer = AdamW
```

The main checkpoint path is:

```text
checkpoints/depth_4_embed_dim_128/best_model.pt
```

Run component ablations with:

```bash
python train_ablation.py
```

Change `CONFIG["ablation_name"]` in `train_ablation.py` to select variants such as `no_layerscale`, `no_droppath`, `mean_pooling`, `pair_aware_pe`, `lag_aware_pe`, or `linear_head`.

## Evaluation

Main paper experiment scripts:

```bash
python exp_diff_models_snr_enhanced.py        # RMSE versus SNR
python exp_diff_models_interval_enhanced.py   # success rate versus angular separation
python exp_diff_models_traversal_enhanced.py  # angular sweep scatter experiments
python exp_diff_models_snapshot.py            # snapshot-number comparison
python exp_nonideal.py                        # mutual coupling and calibration errors
python exp_eff.py                             # parameters, FLOPs, latency, and memory
python exp_diff_models_real.py                # measured mmWave radar data experiment
```

Ablation scripts:

```bash
python exp_diff_heads.py
python exp_diff_mode.py
python exp_diff_loss.py
```

Most scripts keep their experiment constants near the top of the file, including checkpoint path, SNR range, number of Monte Carlo trials, and target angles.

## Checkpoints

The GitHub repository does not include checkpoints or pretrained model weights. These files are ignored by Git to keep the code release lightweight and avoid large binary files in the repository history.

After training or downloading weights separately, place them at the paths expected by the scripts:

```text
checkpoints/depth_4_embed_dim_128/best_model.pt        # main CMA-Former
checkpoints/depth_4_embed_dim_128/*/best_model.pt      # component ablations
checkpoints/heads_*/best_model.pt                      # head-configuration ablations
exp_utils/weights/*.pt, *.pth                          # CNN, DNN, DCT-ViT, HMC-ViT baselines
```

If pretrained checkpoints are released later, it is recommended to host them outside the Git repository, for example through GitHub Releases, Zenodo, Google Drive, or another artifact download link. Generated `DATA/*.h5` datasets should also remain outside Git.

## Citation

If this code is useful for your research, please cite the manuscript:

```bibtex
@article{su2026cmaformer,
  title={Cascaded Multi-Head Attention Transformer Framework for Direction of Arrival Estimation},
  author={Su, Yunye and Wang, Xianpeng and Wen, Linqiang and Guo, Yuehao and Wang, Han and Chen, Gaojie},
  year={2026}
}
```
