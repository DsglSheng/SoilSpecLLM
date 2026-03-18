# Soil Prediction

This folder contains the SoilSpecLLM spectral prediction package adapted from a Time-LLM style backbone for soil VNIR-SWIR reflectance forecasting.

## Task Definition

- Input: preceding reflectance segment in 400-2500 nm
- Output: subsequent reflectance segment to complete the full spectral curve
- Data modality: continuous spectral sequence with prompt-based semantic conditioning

## Method Summary

- A learnable spectral-text alignment module connects spectral tokens and prompt semantics.
- A pretrained LLM embedding space is used as the semantic hub.
- The LLM backbone is used without full retraining, enabling data-efficient transfer.

## Included Code

- `run_main.py`: training and evaluation entry for spectral prediction
- `models/`, `layers/`, `utils/`, `data_provider/`: core model and data pipeline
- `scripts/TimeLLM_Spectral*.sh`: reproducible spectral experiments
- `dataset/prompt_bank/`: prompt templates
- `dataset/spectral/spectral_soil_test.csv`: lightweight example spectral file

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run a baseline spectral prediction experiment:

```bash
bash scripts/TimeLLM_Spectral.sh
```

## Notes

- This release excludes large training libraries and generated visualization artifacts.
- For paper-focused reproducibility, the package keeps only spectral prediction related components.
- Spectral generation components will be added in a separate `Soil generation` folder.

