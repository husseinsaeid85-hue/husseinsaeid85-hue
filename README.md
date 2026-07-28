<h1 align="center">Hussein Said</h1>

<p align="center">
  Machine learning for signals and sensors EEG, EMG, and multispectral imaging.<br>
  I build the whole path: raw sensor data → preprocessing → features → model → real-time inference.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/XGBoost-337AB7?style=for-the-badge" alt="XGBoost">
  <img src="https://img.shields.io/badge/Optuna-1B4F72?style=for-the-badge" alt="Optuna">
  <img src="https://img.shields.io/badge/MNE-006BB6?style=for-the-badge" alt="MNE">
</p>

---

## What I work on

Most of my work sits where **physical sensors meet machine learning**  the messy part, where signals
are noisy, misaligned, and there is never enough labelled data.

Two threads run through it:

**Signals from the body.** Predicting continuous finger force from EEG and EMG recorded
simultaneously aligning three streams at different sampling rates, removing artifacts, extracting
spectral features, and getting a model to run fast enough for real-time inference over a live stream.

**Signals from materials.** Classifying plastic types from near-infrared multispectral imaging, where
polymers that look identical in visible light separate cleanly across nine wavelengths. Comparing deep
learning against gradient boosting on the same problem.

Underneath both: I learned the fundamentals by **building a neural network framework from scratch in
NumPy**  every layer, every backward pass, no autograd.

---

## Research projects

<table>
  <tr>
    <td width="50%" valign="top">
      <h3><a href="https://github.com/husseinsaeid85-hue/eeg-force-prediction">EEG + EMG → Force Prediction</a></h3>
      <p><em>Research internship</em></p>
      <p>Continuous finger force regression from 31-channel EEG and 40-channel high-density EMG.
      Trial-wise alignment via MNE, current source density and ICA artifact removal, multitaper band
      power features, and Optuna-tuned XGBoost and LSTM models — plus a real-time inference path over
      Lab Streaming Layer.</p>
      <p>
        <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
        <img src="https://img.shields.io/badge/MNE-006BB6" alt="MNE">
        <img src="https://img.shields.io/badge/XGBoost-337AB7" alt="XGBoost">
        <img src="https://img.shields.io/badge/Optuna-1B4F72" alt="Optuna">
      </p>
    </td>
    <td width="50%" valign="top">
      <h3><a href="https://github.com/husseinsaeid85-hue/multispectral-plastic-classification">Multispectral Plastic Classification</a></h3>
      <p><em>Master's thesis</em></p>
      <p>Classifying 11 polymer types from near-infrared imaging at 9 wavelengths (800–1350 nm).
      A ResNet-18 with its first convolution rebuilt for 9-channel input, benchmarked against XGBoost
      on spectral intensity features across seven hyperparameter search strategies.</p>
      <p>
        <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
        <img src="https://img.shields.io/badge/ResNet-5C6BC0" alt="ResNet">
        <img src="https://img.shields.io/badge/XGBoost-337AB7" alt="XGBoost">
        <img src="https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white" alt="scikit-learn">
      </p>
    </td>
  </tr>
</table>

---

## Deep learning framework, from scratch

Four repositories, built in sequence a complete neural network framework in pure NumPy with no
autograd, no PyTorch, and every gradient derived and implemented by hand.

```mermaid
flowchart LR
    A["1 · Data<br/>PatternGenDataHandler"] --> B["2 · Core<br/>FullyConnectedNeuralNetwork"]
    B --> C["3 · Convolutional<br/>NeuralNetFramework-CNN"]
    C --> D["4 · Regularization + RNN<br/>Regularization-RecurrentNN"]

    style A fill:#0d47a1,stroke:#64b5f6,color:#fff
    style B fill:#1565c0,stroke:#64b5f6,color:#fff
    style C fill:#1976d2,stroke:#64b5f6,color:#fff
    style D fill:#1e88e5,stroke:#64b5f6,color:#fff
```

| # | Repository | What it adds |
|---|---|---|
| 1 | [PatternGenDataHandler](https://github.com/husseinsaeid85-hue/PatternGenDataHandler) | Pattern generation and an augmenting image batch loader |
| 2 | [FullyConnectedNeuralNetwork](https://github.com/husseinsaeid85-hue/FullyConnectedNeuralNetwork) | Fully connected layers, ReLU, SoftMax, cross-entropy, SGD |
| 3 | [NeuralNetFramework-CNN](https://github.com/husseinsaeid85-hue/NeuralNetFramework-CNN) | Conv 1D/2D, pooling, flatten, Xavier/He init, Adam and momentum |
| 4 | [Regularization-RecurrentNN](https://github.com/husseinsaeid85-hue/Regularization-RecurrentNN) | L1/L2 regularization, dropout, batch norm, RNN |

---

## Toolbox

| | |
|---|---|
| **Languages** | Python |
| **Deep learning** | PyTorch, torchvision, transfer learning, CNNs, LSTMs, RNNs |
| **Classical ML** | scikit-learn, XGBoost, PCA, cross-validation |
| **Tuning** | Optuna, Bayesian optimization, grid and randomized search |
| **Signals** | MNE, ICA, current source density, multitaper spectral estimation, Savitzky–Golay filtering |
| **Data** | NumPy, pandas, matplotlib, seaborn |
| **Infrastructure** | HPC clusters (SLURM), Lab Streaming Layer, uv, git |

---

<p align="center">
  <img src="https://github-readme-stats.vercel.app/api?username=husseinsaeid85-hue&show_icons=true&hide_border=true&theme=transparent&hide=stars" alt="GitHub stats" height="150">
  <img src="https://github-readme-stats.vercel.app/api/top-langs/?username=husseinsaeid85-hue&layout=compact&hide_border=true&theme=transparent" alt="Top languages" height="150">
</p>
