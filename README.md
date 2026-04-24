# Am_Eu_ligand_design

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A closed-loop molecular design framework integrating **JT-VAE**, **XGBoost**, and **PPO** for discovering high-selectivity ligands for Am(III)/Eu(III) separation in nuclear fuel reprocessing.

This repository accompanies the paper: *AI-Guided Ligand Design for Am(III)/Eu(III) Separation* (2025).

---

## Overview

Efficient separation of Am(III) from Eu(III) is a key challenge in nuclear waste management due to their nearly identical chemical properties. This framework combines:

- **JT-VAE**: Learns a continuous latent representation of metal–organic ligands from >44,000 CSD complexes (99.44% generation validity).
- **XGBoost**: Predicts the Am/Eu separation factor (*SF*) from molecular fingerprints and experimental conditions (test R² = 0.8663).
- **PPO**: Performs goal-directed optimization in latent space to discover novel high-*SF* ligands.

**Key result**: 80% of literature ligands improved after optimization, with an average SF increase of **35.6%**.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/CainLiTian/Am_Eu_ligand_design.git
cd Am_Eu_ligand_design

# Create a conda environment (recommended)
conda create -n ameu_ligand python=3.9
conda activate ameu_ligand

# Install dependencies
pip install -r requirements.txt
