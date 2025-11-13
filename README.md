<div align="center">
<h1> MedSpikeFormer: All-Neuron Decomposed Self-Attention for Medical Image Segmentation </h1>
</div>

## 🎈 News

Training and testing code released.



## ⭐ Abstract

Spiking self-attention (SSA) has emerged as a promising approach for medical image segmentation. However, segmentation performance still degrades in complex scenarios where salient and non-salient regions coexist. Two fundamental issues remain: i) existing SSA mechanisms rely only on \emph{activated} neurons, overlooking the contextual cues carried by \emph{inactivated} neurons, and ii) the binary spike representation causes distribution distortions that make spiking self-attention lag behind their ANN-based self-attention in spatial discriminability. To overcome these challenges, we propose MedSpikeFormer, a spiking transformer that explicitly integrates both activated and inactivated neurons via an All-Neuron Decomposed Self-Attention (AND-SA). Specifically, AND-SA explicitly models four types of neuronal interactions: activated--activated, activated--inactivated, inactivated--activated, and inactivated--inactivated, thus recovering rich contextual dependencies ignored by conventional SSA. Plus, we employ a distribution alignment loss that minimizes the divergence between AND-SA and ANN-based self-attention while preserving the binary nature of spiking neural networks, significantly improving spatial feature discriminability. Experiments on 5 medical segmentation benchmarks show that MedSpikeFormer outperforms 14 methods, achieving up to +2.4\% mIoU on ISIC2018 and +8.7\% on COVID-19. These results confirm that leveraging both fired and non-fired neurons is crucial for robust spike-driven medical image segmentation.

## 🚀 Introduction

<div align="center">
    <img width="800" alt="image" src="Test/figures/challenge.png?raw=true">
</div>

The challenges: the misleading co-occurrence of salient and non-salient objects. SA is ANN-based self-attention and SSA denotes spike self-attention.

## 📻 Overview

<div align="center">
<img width="800" alt="image" src="Test/figures/network.png?raw=true">
</div>

Illustration of the overall architecture.


## 📆 TODO

- [x] Release code

## 🎮 Getting Started

### 1. Install Environment

```
See requirements
```

### 2. Prepare Datasets

- Download datasets: Moun-Seg from this [link](https://www.kaggle.com/datasets/tuanledinh/monuseg2018).

### 3. Run the Net

```
Train the Net:  python train.py
```

```
Test the Net:  python test.py
```

## 🖼️ Visualization Comparison

<div align="center">
<img width="800" alt="image" src="Test/figures/com_pic.png?raw=true">
</div>

<div align="center">
We compare our method against 14 methods. The red box indicates the area of incorrect predictions.
</div>

## ✨ Quantitative Comparison

<div align="center">
<img width="800" alt="image" src="Test/figures/com_tab.png?raw=true">
</div>

<div align="center">
Performance comparison with 14 methods on 5 datasets.
</div>

## ⭐ Statistical Significance

<div align="center">
<img width="200" alt="image" src="Test/figures/com_ss.png?raw=true">
</div>

<div align="center">
Paired t-test p-values comparing our method with other SOTAs.
</div>


## 🖼️ Visualization of Ablation Results

<div align="center">
<img width="800" alt="image" src="Test/figures/aba.png?raw=true">
</div>

<div align="center">
Ablation Visualization comparison. The red box indicates the area of incorrect predictions.
</div>

## 🖼️ Convergence Analysis

<div align="center">
<img width="800" alt="image" src="Test/figures/curve.png?raw=true">
</div>

<div align="center">
The Mean Intersection over Union (mIoU) curves.
</div>
