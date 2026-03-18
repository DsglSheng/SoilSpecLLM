# SoilSpecLLM

SoilSpecLLM is a soil VNIR-SWIR spectral learning framework for 400-2500 nm reflectance modeling.
The framework aligns soil-property semantics with continuous spectral signals and supports two tasks:

- spectral prediction: predict missing/following spectral segments from preceding bands
- spectral generation: condition spectral synthesis on soil property descriptions

This repository version is organized for paper code release and currently focuses on the spectral prediction pipeline.

## Research Scope

The method is designed for global-scale soil spectral data and targets:

- accurate full-curve reconstruction/prediction
- robust cross-sensor generalization
- physically consistent absorption behavior around 1.4, 1.9, and 2.2 um

## Keywords

- Soil spectrum
- Large language model
- Diffusion model

## Repository Layout

- `Soil Prediction/`: complete spectral prediction code package used for this paper release.
- `Soil Generation/`: diffusion-based spectral generation code package for soil spectrum synthesis.
