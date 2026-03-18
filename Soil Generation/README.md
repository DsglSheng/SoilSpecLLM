# Soil Generation

This folder contains the spectral generation pipeline used by SoilSpecLLM.

## Task

Generate full soil reflectance spectra (400-2500 nm) from semantic soil-property embeddings and optional structured conditions.

## Core Components

- `train_soil_generation.py`: train diffusion model in latent space
- `infer_soil_generation.py`: generate spectra and optional uncertainty analysis
- `config/soil_generation.json`: training/inference configuration template
- `dataset/soil_gen_dataset.py`: `DictDataset` loader for latent+label dictionaries
- `unet/`: conditional and non-conditional U-Net backbones
- `utils/train.py`: diffusion training loop
- `utils/inference_batch_png.py`: spectrum generation and evaluation visualization
- `vae/vae.py`: latent decoder to reconstruct full spectra

## Data Format (expected by DictDataset)

Each sample in the `.pt` dictionary should include:

- `data`: latent tensor (e.g., `[4, 263]`)
- `label`: dictionary with at least `text_embed`, optionally `raw_spectra` and soil properties

## Usage

```bash
python train_soil_generation.py --config config/soil_generation.json
python infer_soil_generation.py --config config/soil_generation.json --uncertainty --uncertainty_runs 10
```

## Notes

- This release keeps only code relevant to soil spectral generation.
- Model weights, large datasets, and generated images are intentionally excluded.

