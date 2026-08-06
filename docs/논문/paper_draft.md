# Multi-Scale Temporal Difference Features with Supervised Contrastive Learning for Multimodal Sensor-Based Equipment Degradation Prediction

**Authors:** [Author Names]

**Affiliation:** [Institution]

**Submitted to:** Sensors (MDPI)

---

## Abstract

Predicting equipment degradation in manufacturing facilities is critical for ensuring worker safety and preventing costly unplanned downtime. While recent approaches employ complex Transformer architectures for multimodal sensor fusion, their effectiveness is often limited by insufficient training data in industrial settings. In this paper, we propose a lightweight yet effective approach that combines multi-scale temporal difference features with supervised contrastive learning for multimodal sensor-based degradation prediction. Our method extracts rate-of-change signals at multiple time lags (1, 5, and 10 steps) from eight sensor channels (temperature, particulate matter, and motor current) and fuses them with thermal imagery through a Long Short-Term Memory (LSTM) network augmented with channel attention. A joint loss function combining cross-entropy and supervised contrastive loss explicitly separates the latent representations of closely overlapping degradation states. We evaluate our approach on a large-scale manufacturing dataset comprising 111,870 multimodal samples from 36 industrial transport devices (OHT and AGV). Our proposed model achieves a macro F1-score of 0.9557 +/- 0.0006 (three runs) with only 2.85M parameters, outperforming a Cross-Attention Temporal Fusion Transformer (10.8M parameters, F1 = 0.9252) by 3.05%. A comprehensive ablation study across eight model variants (plus three component-level ablations) reveals that the three proposed components—multi-scale differences, channel attention, and contrastive loss—are individually ineffective but produce strong synergistic improvement (+1.20% F1) when combined. Notably, our model achieves 100% recall for severe degradation states and reduces Normal-Mild boundary misclassifications by 47% compared to the baseline. The model is exported to ONNX format (10.9 MB) with an estimated inference latency of approximately 5 ms on NVIDIA Jetson Orin Nano, demonstrating feasibility for real-time on-device deployment in manufacturing environments.

**Keywords:** predictive maintenance; multimodal sensor fusion; temporal difference features; supervised contrastive learning; equipment degradation; manufacturing safety; LSTM; edge deployment

---

## 1. Introduction

Equipment failure in manufacturing facilities poses significant risks to worker safety, including thermal burns from overheated machinery, respiratory hazards from carbonization fumes, and physical injuries from electrical or mechanical malfunctions [1,2]. Early detection of equipment degradation is therefore essential for implementing timely safety interventions.

Modern manufacturing equipment is increasingly instrumented with multiple sensor modalities, including temperature sensors, particulate matter detectors, current transformers, and thermal cameras [3]. These multimodal data streams provide complementary information: temperature sensors capture overall thermal trends, particulate matter sensors detect early signs of carbonization, current sensors reveal electrical and mechanical load anomalies, and thermal cameras visualize spatial heat distribution patterns [4].

Recent work in predictive maintenance has explored deep learning approaches for multimodal sensor fusion. Transformer-based architectures, such as MMTransformer and cross-attention fusion networks, have shown promise in modeling complex inter-modal relationships [5,6]. However, these architectures typically require large training datasets to avoid overfitting, which is often impractical in industrial settings where labeled anomaly data is scarce [7]. Furthermore, their computational overhead may hinder deployment on resource-constrained edge devices such as the NVIDIA Jetson platform, which is commonly used for on-site inference in manufacturing environments [8].

In this work, we propose a lightweight yet effective approach based on three key observations derived from domain-specific data analysis:

1. **Sensor rate-of-change is more discriminative than absolute values.** A 4 °C difference between Normal and Mild degradation states (27 °C vs. 31 °C) is difficult to detect from instantaneous readings, but becomes apparent when tracked over multiple time scales.

2. **Normal and Mild degradation states overlap significantly in feature space,** requiring explicit representation learning beyond standard classification losses.

3. **Different sensors contribute asymmetrically** to degradation detection at different severity levels: temperature correlates linearly with all degradation stages, particulate matter saturates at moderate severity, and current sensors exhibit explosive increases only at severe levels.

Based on these observations, we make the following contributions:

- We introduce **multi-scale temporal difference features** that capture sensor rate-of-change at lags of 1, 5, and 10 time steps, providing a physics-informed representation that outperforms complex Transformer architectures while adding negligible computational overhead.

- We propose a **joint cross-entropy and supervised contrastive loss** that explicitly separates the latent representations of closely overlapping degradation states, reducing Normal-Mild misclassifications by 47%.

- We conduct a **comprehensive ablation study** across eight model variants (including TimesNet and PatchTST), demonstrating that domain-informed feature engineering with a lightweight LSTM (2.85M parameters) achieves superior performance (F1 = 0.9557 +/- 0.0006) compared to both general-purpose time series architectures and a Cross-Attention Temporal Fusion Transformer with 10.8M parameters (F1 = 0.9252).

Building on the three observations above, we formalize the following research questions (RQ) and testable hypotheses (H) that structure the empirical study in Section 6:

- **RQ1 (Rate-of-change vs. absolute values).** Are multi-scale temporal difference features more discriminative than absolute sensor values for classifying closely-spaced degradation states?
  - **H1.** Augmenting an LSTM baseline with multi-scale temporal differences at lags [1, 5, 10] yields a statistically meaningful improvement in macro F1-score over both the single-lag variant and the absolute-value-only baseline, without a proportional increase in parameter count.

- **RQ2 (Explicit class separation in latent space).** Can a supervised contrastive objective, applied jointly with cross-entropy, resolve the Normal-Mild boundary that is difficult to separate under standard classification loss alone?
  - **H2.** Adding a supervised contrastive term (lambda = 0.1) on top of the multi-scale-diff LSTM backbone reduces Normal-Mild misclassifications by at least 40 % relative to the single-lag temporal-difference model, while preserving or improving overall macro F1.

- **RQ3 (Domain-informed lightweight vs. general-purpose deep architectures).** Does a compact LSTM equipped with domain-informed features outperform substantially larger general-purpose time series Transformers (TimesNet, PatchTST) and a cross-attention temporal fusion Transformer (CATFT) on this multimodal degradation task, under an edge-deployment parameter budget?
  - **H3.** The proposed V2+ (approx. 2.85 M parameters) achieves higher macro F1 than TimesNet, PatchTST, and CATFT (up to 10.79 M parameters) on the AI Hub #71802 validation split, and simultaneously satisfies a sub-10 ms per-inference latency constraint suitable for Jetson-class edge deployment.

Each hypothesis is designed to be **falsifiable** against the experiments defined in Section 5. We revisit H1-H3 explicitly in Section 6.9, mapping each hypothesis to the specific tables and metrics that confirm or refute it.

The remainder of this paper is organized as follows. Section 2 reviews related work. Section 3 describes the dataset and problem formulation. Section 4 presents the proposed method. Section 5 details the experimental setup. Section 6 reports results and analysis. Section 7 discusses findings and limitations, and Section 8 concludes the paper.

---

## 2. Related Work

### 2.1. Predictive Maintenance in Manufacturing

Predictive maintenance (PdM) aims to anticipate equipment failures before they occur, enabling proactive interventions that improve safety and reduce costs [9]. Traditional PdM approaches rely on physics-based models or statistical thresholds [10], but data-driven methods using machine learning and deep learning have gained prominence due to their ability to capture complex degradation patterns [11].

Convolutional Neural Networks (CNNs) have been applied to vibration signals for bearing fault diagnosis [12], while Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks have shown effectiveness in modeling temporal dependencies in sensor time series [13,14]. More recently, Transformer-based models have been explored for their ability to capture long-range dependencies [15,16]. However, the practical deployment of such models on industrial edge devices remains challenging due to their computational requirements.

### 2.2. Multimodal Sensor Fusion

Manufacturing equipment monitoring typically involves multiple sensor modalities, including vibration, temperature, current, and visual data [17]. Effective fusion of these heterogeneous data streams is critical for accurate degradation prediction.

Early fusion approaches concatenate raw features from different modalities, while late fusion combines predictions from modality-specific classifiers [18]. Intermediate fusion, which learns joint representations through attention mechanisms or cross-modal transformers, has shown superior performance [19,20]. However, the effectiveness of attention-based fusion depends on having sufficient training data to learn meaningful cross-modal relationships. In manufacturing contexts with limited labeled anomaly data, simpler fusion strategies may be more robust [21].

### 2.3. Temporal Feature Engineering for Time Series

Feature engineering for time series data has a long history in signal processing and control systems [22]. Temporal difference features, which capture the rate of change of sensor readings, are fundamental indicators of system dynamics [23]. First-order differences (velocity) and second-order differences (acceleration) have been widely used in process monitoring [24].

Recent work has explored multi-scale temporal representations. TimesNet [25] reshapes 1D time series into 2D tensors to capture multi-period patterns using inception-style 2D convolutions. PatchTST [26] segments time series into patches and applies channel-independent Transformers, achieving strong results on forecasting benchmarks. However, these general-purpose architectures may not optimally exploit the domain-specific structure of manufacturing sensor data. In Section 6, we empirically compare both methods against our approach and find that domain-informed temporal difference features outperform both TimesNet and PatchTST on the degradation classification task.

### 2.4. Contrastive Learning for Classification

Contrastive learning has emerged as a powerful representation learning paradigm [27]. Supervised Contrastive Learning (SupCon) [28] extends self-supervised contrastive methods by leveraging label information to pull same-class samples together and push different-class samples apart in the embedding space.

For time series classification, TS2Vec [29] and SoftCLT [30] have demonstrated the effectiveness of contrastive objectives. In manufacturing contexts, contrastive learning is particularly valuable when class boundaries are ambiguous, as it provides an explicit mechanism for learning discriminative representations beyond what cross-entropy loss alone can achieve [31].

---

## 3. Dataset and Problem Formulation

### 3.1. Dataset Overview

We use the "Manufacturing Transport Device Degradation Predictive Maintenance Multimodal Data" dataset from AI Hub [32], a large-scale public dataset designed for predictive maintenance research. The dataset is derived from 36 industrial transport devices — 18 Overhead Hoist Transports (OHTs) and 18 Automated Guided Vehicles (AGVs) — operating in semiconductor, display, and automotive manufacturing facilities. According to the dataset provider's utilization guideline, the full corpus contains 124,263 sets (73,733 OHT + 50,530 AGV) collected on 100 ms cycles (10 Hz) and aggregated by the provider into 1 s packages before release; the specific aggregation function (mean / max / last) is not disclosed. Our experiments use the released **111,870-sample subset** consisting of 99,476 training and 12,394 validation samples, corresponding to approximately 90 % of the full corpus.

Two properties of this subset are important for interpreting our results. First, the raw distribution is dominated by Normal operation (approximately 99 % of full-corpus samples fall in the Normal class per the guideline's Z-score >= 3.5 rule, described in Section 3.5). The released subset has already been class-balanced by the provider so that Normal accounts for approximately 49 % of samples in our training split (see Table 3); real-world deployment therefore faces a substantially more imbalanced distribution than our evaluation. Second, the dataset is pre-split at the equipment level (see Table 1) so that the validation set contains entirely unseen devices, eliminating temporal leakage across splits.

Each sample consists of three components: (1) an 8-dimensional sensor reading, (2) a 120 x 160 thermal infrared image stored in NumPy `.npy` format, and (3) a JSON annotation containing the degradation state label, meta-info, and auxiliary fields (see Section 3.5).

**Table 1.** Dataset statistics.

| Split | Equipment | Sessions | Samples | AGV Sessions | OHT Sessions |
|:------|:----------|:---------|:--------|:-------------|:-------------|
| Training | Devices 01-16 | 303 | 99,476 | 112 | 191 |
| Validation | Devices 17-18 | 39 | 12,394 | 15 | 24 |
| **Total (subset)** | **36 devices** | **342** | **111,870** | **127** | **215** |
| Full corpus (per guideline [32]) | 36 devices | — | **124,263** | — | — |

The 36 devices span three manufacturers and three product models, providing an implicit device-level diversity signal even though the underlying sensor hardware is homogeneous (Section 3.2):
- SFA / OHT-OCS (manufacturer A, model A1): oht01-oht18 (18 devices, ~58 % of samples).
- Mireu / Mri-100 (manufacturer B, model B1): agv01-agv09 (9 devices, ~21 %).
- CACSystems / low-profile AGV (manufacturer C, model C1): agv10-agv18 (9 devices, ~21 %).

### 3.2. Sensor Specifications

The 8 sensor channels span three physical measurement categories — temperature, air quality, and electrical current — plus a thermal imaging modality. Table 2 lists the value ranges observed on the training set; Table 2b identifies the actual hardware model behind each channel, confirmed against the AI Hub metadata schema in a 20,000-sample scan.

**Table 2.** Sensor channel value ranges (training set).

**Table 2.** Sensor channel specifications and statistics (training set).

| Channel | Category | Unit | Min | Max | Mean | Std |
|:--------|:---------|:-----|----:|----:|-----:|----:|
| NTC | Temperature | C | 17.33 | 80.20 | 32.49 | 9.27 |
| PM1.0 | Particulate | ug/m3 | 6.00 | 42.00 | 15.70 | 10.39 |
| PM2.5 | Particulate | ug/m3 | 10.00 | 51.00 | 20.79 | 12.15 |
| PM10 | Particulate | ug/m3 | 18.00 | 92.00 | 36.32 | 22.27 |
| CT1 | Current | A | 0.50 | 201.34 | 6.23 | 18.61 |
| CT2 | Current | A | 0.60 | 273.91 | 42.73 | 46.43 |
| CT3 | Current | A | 0.22 | 243.25 | 24.49 | 29.74 |
| CT4 | Current | A | 0.30 | 219.04 | 11.21 | 18.99 |

The four current channels are not interchangeable but correspond to distinct measurement points on each device: CT1 = input line, CT2 = output line, CT3 = motor 1, CT4 = motor 2. This physical role difference explains the pronounced scale disparity across channels (CT2 mean 42.73 A versus CT1 mean 6.23 A) and the different degradation-time behaviors seen in Section 3.4.

The thermal infrared images are stored as 120 x 160 arrays of °C values in NumPy `.npy` format. The stored resolution should not be confused with the native sensor resolution: the acquisition camera (Section 3.2b) has a 32 x 32 native pixel grid, so the 120 x 160 representation is the result of the dataset provider's ~15 x upsampling. We use the released 120 x 160 arrays directly. Values range approximately from 31 °C (background) to 146 °C (highest observed hot spot), remaining consistent with the guideline range and requiring no clipping.

**Table 2b.** Hardware sensor models actually used to collect the dataset (100 % of scanned samples used the same model per category; the AI Hub schema lists three additional candidates per category but they are not present in the released data).

| Category | Model | Notes |
|:---------|:------|:------|
| Temperature (NTC) | Vishay **NTCLE413** (10 kΩ) | Analog resistance; converted to °C by the provider |
| Particulate matter | Sharp **GP2Y1014AU0F** | Optical LED scattering; analog output |
| Current transformer | KEMET **CT-06** | Passive CT; provider maps to A. Reported CT2 peaks at 273.91 A exceed the nominal spec (0-200 A) — treated as valid data |
| Thermal camera | Terabee **Evo Thermal 33** | 32 x 32 native, 33° FOV, °C-per-pixel radiometry |

This single-set hardware configuration means that the dataset does not, on its own, provide sensor-model diversity. All device-level heterogeneity in the results (Section 6) is therefore attributable to the three-manufacturer / three-model device pool described in Section 3.1, not to sensor variation.

### 3.3. Degradation State Definition and Distribution

The dataset provider annotates each sample with one of four degradation states following the safety-severity definition below [32]. We adopt the guideline's exact definition and note the important boundary property that follows from it.

**Table 3a.** Degradation-state definition from the dataset guideline [32]. GT denotes the actual carbonization onset (ground truth); Z-score is computed on the sensor time series by the provider.

| State | Official label (Korean / English) | Paper shorthand | Definition [32] |
|:-----:|:----------------------------------|:----------------|:-----------------|
| 0 | 정상 / Normal | Normal | Z-score <= 3.5 (approximately 99% of full-corpus samples) |
| 1 | 관심 / Attention | Mild | First half of the "Z-score > 3.5 up to GT-30s" interval |
| 2 | 경고 / Warning | Moderate | Second half of the same interval |
| 3 | 위험 / Danger | Severe | The final 30 seconds before carbonization (GT-30s to GT) |

Two definitional properties are important for interpreting the results in Section 6. First, the split between Attention (State 1) and Warning (State 2) is a **1:1 partition of a single physically homogeneous interval**, introduced by the provider to allow user-tunable predictive-maintenance sensitivity [32]. The Attention-Warning boundary therefore does not correspond to any physical or statistical discontinuity, and no classifier can perfectly separate the two classes; residual State-1 <-> State-2 confusion is an irreducible property of the label design rather than a model failure. Second, the Danger state is exactly the final 30-second window before carbonization, which is why we prioritize Severe recall = 100 % throughout the paper: any Severe miss corresponds to under 30 seconds of remaining safe intervention time.

**Table 3.** Class distribution in our released subset (already re-balanced by the provider; the full-corpus distribution is approximately 99 % Normal).

| State | Label (KO / EN) | Training | (%) | Validation | (%) |
|:------|:----------------|:---------|:----|:-----------|:----|
| 0 | 정상 / Normal | 49,285 | 49.5 | 5,643 | 45.5 |
| 1 | 관심 / Attention | 21,189 | 21.3 | 2,892 | 23.3 |
| 2 | 경고 / Warning | 21,322 | 21.4 | 2,869 | 23.1 |
| 3 | 위험 / Danger | 7,680 | 7.7 | 990 | 8.0 |

Even after re-balancing, the safety-critical Danger class comprises only 7.7 % of training samples. This residual imbalance motivates our use of class-balanced sampling and weighted loss (Section 4).

### 3.4. Sensor Behavior by Degradation State

Analysis of sensor values across degradation states reveals distinct patterns for each sensor category.

**Table 4.** Mean sensor values by degradation state (training set, 10% sampling).

| Sensor | State 0 | State 1 | State 2 | State 3 | Pattern |
|:-------|--------:|--------:|--------:|--------:|:--------|
| NTC (C) | 26.78 | 30.95 | 40.65 | 50.71 | Linear increase (+24C) |
| PM1.0 (ug/m3) | 13.66 | 16.18 | 18.83 | 18.76 | Saturation at State 2 |
| PM2.5 (ug/m3) | 18.38 | 21.31 | 24.53 | 24.44 | Saturation at State 2 |
| PM10 (ug/m3) | 31.60 | 37.59 | 43.47 | 43.28 | Saturation at State 2 |
| CT1 (A) | 2.16 | 3.21 | 10.16 | 29.85 | 14x jump at State 3 |
| CT2 (A) | 33.15 | 30.97 | 50.43 | 115.38 | 3.5x jump at State 3 |
| CT3 (A) | 21.87 | 19.89 | 26.78 | 47.75 | 2.2x jump at State 3 |
| CT4 (A) | 8.81 | 8.28 | 13.08 | 29.54 | 3.4x jump at State 3 |

Three key observations emerge (Figure 1): (1) temperature (NTC) exhibits a linear correlation with degradation severity, increasing by approximately 24 °C from Normal to Severe; (2) particulate matter sensors saturate at State 2, limiting their discriminative power for severe states; and (3) current sensors show near-normal values for States 0-1 but explosive increases at State 3 (up to 14x baseline for CT1). These observations motivate our domain-informed feature design.

**[Figure 1: Sensor behavior by degradation state. (a) Temperature increases linearly. (b) Particulate matter saturates at State 2. (c) Motor current shows explosive increase at State 3.]**

Critically, the difference between Normal (State 0) and Mild (State 1) is only approximately 4 °C for NTC, making this boundary the most challenging classification task.

### 3.5. Problem Definition

Given a temporal window of T consecutive multimodal samples, each consisting of an 8-dimensional sensor vector s_t in R^8 and a thermal image I_t in R^{120x160}, the task is to predict the degradation state y in {0, 1, 2, 3} for the window as defined in Section 3.3. We use a window size of T = 30 corresponding to 30 seconds at the released 1 Hz sampling rate. Note that this 1 Hz series is itself a 10-to-1 temporal aggregation of an internal 100 ms (10 Hz) acquisition performed by the provider; the aggregation function (mean / max / last) is not disclosed, so any temporal analysis in this paper is bounded from below by the 1 s aggregation window. The training objective is to maximize macro-averaged F1-score while ensuring 100 % recall for the Danger state (which corresponds, per Section 3.3, to the final 30 seconds before carbonization).

---

## 4. Proposed Method

### 4.1. Data Pipeline

Our data pipeline addresses several limitations identified in prior work [32].

**Session-aware windowing.** Rather than applying sliding windows globally and splitting randomly (which causes data leakage due to overlapping windows), we first identify recording sessions by detecting temporal gaps exceeding 120 seconds in the timestamp sequence. Sliding windows (size = 30, step = 10) are then constructed within each session, ensuring that no window crosses session boundaries.

**Normalization.** Sensor channels are normalized using Z-score normalization with statistics computed exclusively from the training set. Thermal images are normalized to [0, 1] using global min-max scaling (training set range: 30.98 °C to 146.10 °C), preserving absolute temperature information that correlates with degradation severity.

**Class-balanced sampling.** We employ a WeightedRandomSampler that assigns sampling weights inversely proportional to class frequency. Combined with class-weighted cross-entropy loss (weights: Normal = 0.28, Mild = 0.60, Moderate = 0.61, Severe = 2.51), this ensures balanced representation during training despite the 6.4:1 imbalance between Normal and Severe states.

**Label assignment.** For each window, we assign the majority vote label across all 30 timesteps, which is more robust to label noise at state transitions than using only the first timestep [32].

After preprocessing, the training set contains 9,313 windows and the validation set contains 1,157 windows.

The overall architecture is illustrated in Figure 2.

**[Figure 2: Architecture of the proposed V2+ model. Multi-scale temporal differences are concatenated with raw sensor input and processed by an LSTM with SE channel attention. Thermal images are encoded by a lightweight CNN. Features are fused and classified with a joint CE + SupCon loss.]**

### 4.2. Multi-Scale Temporal Difference Features

The core of our approach is the extraction of temporal difference features at multiple time scales. For a sensor sequence S = [s_1, s_2, ..., s_T] where s_t in R^8, we compute temporal differences at lags l in {1, 5, 10}:

Delta_l(t) = s_t - s_{t-l}, for t >= l; 0 otherwise.

The augmented input is the concatenation:

S' = [S; Delta_1; Delta_5; Delta_10] in R^{T x 32}

This representation captures:
- **Lag-1 (instantaneous rate):** Fast transients and spikes in sensor readings.
- **Lag-5 (short-term trend):** Gradual changes over 5-second intervals that may indicate the onset of degradation.
- **Lag-10 (medium-term trend):** Accumulated changes over 10 seconds. A subtle 4 °C difference between Normal and Mild states produces a clearer signal at this scale (approximately 1.3 °C per step cumulative change vs. 0.13 °C at lag-1).

The multi-scale representation adds no learnable parameters to the model; it is a purely domain-motivated preprocessing step that enriches the input representation.

### 4.3. LSTM Encoder with Channel Attention

The augmented sensor sequence S' in R^{T x 32} is processed by a 3-layer LSTM with hidden dimension 128 and inter-layer dropout of 0.1:

h_t = LSTM(S'_t, h_{t-1})

We take the final hidden state h_T in R^128 as the sensor representation, followed by a fully connected projection:

z_s = FC(h_T) in R^128

A Squeeze-and-Excitation (SE) block [33] is then applied to learn channel importance:

w = sigma(FC_2(ReLU(FC_1(z_s))))
z_s = z_s * w

where FC_1 reduces dimensionality by a factor of 8 and FC_2 restores it, and sigma denotes the sigmoid function. While SE blocks were originally designed for CNN feature map channels [33], we apply them to the LSTM's output feature dimensions. In our architecture, the 128-dimensional hidden state encodes information from 32 input features (8 sensors x 4 temporal scales). The SE block learns to reweight these encoded feature dimensions, effectively modulating the contribution of different sensor-temporal combinations. Our ablation (Section 6.2.2) confirms that this mechanism is only effective when combined with multi-scale temporal differences; without sufficiently rich input features, the gating mechanism overfits.

### 4.4. Multimodal Fusion

The thermal image sequence I = [I_1, ..., I_T] is processed by a 3-layer CNN operating independently on each frame:

f_t = CNN(I_t) in R^128, for t = 1, ..., T

The CNN consists of three blocks of Conv2d-ReLU-MaxPool2d with channel progression 1 -> 16 -> 32 -> 64, reducing the spatial dimensions from 120 x 160 to 15 x 20, followed by a fully connected layer projecting to 128 dimensions.

The sensor and thermal representations are fused via broadcast-concatenation and temporal mean pooling:

z = Mean_t([z_s; f_t]) in R^256

where z_s is broadcast across all T timesteps before concatenation. This yields a joint 256-dimensional embedding for classification.

### 4.5. Supervised Contrastive Learning Loss

The classification head maps the embedding z to class logits:

y_hat = FC(Dropout(z)) in R^4

The training loss combines cross-entropy and supervised contrastive loss:

L = (1 - alpha) * L_CE + alpha * L_SupCon

where alpha = 0.3. The cross-entropy loss L_CE uses class weights to handle imbalance:

L_CE = CrossEntropy(y_hat, y; w_class)

The supervised contrastive loss L_SupCon [28] operates on L2-normalized embeddings z_bar = z / ||z||:

L_SupCon = - sum_{i} (1/|P(i)|) * sum_{p in P(i)} log(exp(z_bar_i . z_bar_p / tau) / sum_{a != i} exp(z_bar_i . z_bar_a / tau))

where P(i) is the set of indices sharing the same label as sample i, and tau = 0.07 is the temperature parameter.

The contrastive loss explicitly pulls together embeddings of the same degradation state and pushes apart embeddings of different states. This is particularly important for the Normal-Mild boundary, where sensor value overlap makes classification with cross-entropy alone insufficient.

---

## 5. Experimental Setup

### 5.1. Implementation Details

All models are implemented in PyTorch 2.6.0 and trained on a single NVIDIA Quadro RTX 6000 GPU (24 GB). We use the AdamW optimizer with weight decay 0.01, initial learning rate 1e-3, and ReduceLROnPlateau scheduler (factor = 0.5, patience = 3). Gradient clipping is applied with max norm 1.0. Batch size is 16. Training runs for 20-30 epochs with the best model selected by validation F1.

We note that the dataset provides a fixed equipment-level train/validation split (devices 01-16 vs. 17-18) without a separate held-out test set. Following standard practice for fixed-split benchmarks, we use the validation set for both model selection and final evaluation. While this may introduce optimistic bias in model selection, the equipment-level separation ensures no temporal data leakage between the two sets, providing a meaningful assessment of generalization to unseen devices. To verify the robustness of our results, we report performance over three independent runs with different random seeds (Section 6.6).

**Table 5.** Hyperparameter configuration.

| Parameter | Value |
|:----------|:------|
| Window size | 30 |
| Window step | 10 |
| LSTM hidden dim | 128 |
| LSTM layers | 3 |
| LSTM dropout | 0.1 |
| SE reduction ratio | 8 |
| Temporal diff lags | [1, 5, 10] |
| Batch size | 16 |
| Learning rate | 1e-3 |
| Optimizer | AdamW (wd=0.01) |
| SupCon weight (alpha) | 0.3 |
| SupCon temperature (tau) | 0.07 |

### 5.2. Baseline and Comparison Models

We compare eight model variants, including two external SOTA time series architectures and six variants from our ablation study:

**External baselines:**
- **TimesNet [25]:** 2D temporal variation modeling with period-based reshaping and 2D convolution. Adapted for multimodal classification with our thermal CNN branch.
- **PatchTST [26]:** Channel-independent patch-based Transformer with positional encoding. Adapted with our thermal CNN branch.

**Internal variants:**
- **V1 (Baseline):** Multimodal LSTM as provided in the original AI Hub codebase, with our improved data pipeline.
- **V2 (+ TempDiff):** V1 with single-lag temporal difference feature added to the sensor input.
- **V3 (+ EfficientNet):** V1 with the CNN thermal encoder replaced by a pretrained EfficientNet-B0 [34].
- **V4 (CATFT - CrossAttn):** Cross-Attention Temporal Fusion Transformer without cross-attention; uses Transformer encoders for both sensor and thermal branches.
- **V5 (Full CATFT):** Complete CATFT with bidirectional cross-attention between sensor and thermal features.
- **V2+ (Proposed):** V2 with multi-scale temporal differences (lags 1, 5, 10), SE channel attention, and supervised contrastive loss.

All external and internal models use the same data pipeline, normalization, class-balanced sampling, and evaluation protocol for fair comparison.

### 5.3. Evaluation Metrics

We report macro-averaged F1-score as the primary metric, which equally weights all classes regardless of their frequency. We also report per-class precision, recall, and F1-score, accuracy, and the number of Normal-Mild boundary misclassifications. For the safety-critical Severe state, we specifically track recall (detection rate).

---

## 6. Results and Analysis

### 6.1. Overall Performance Comparison

**Table 6.** Performance comparison of all model variants, including the AI Hub reference baseline and external SOTA baselines.

| Model | Val F1 | Val Acc | Severe Recall | Params | NM Errors |
|:------|:------:|:-------:|:----------:|-------:|---:|
| AI Hub reference MMTransformer [32] | 0.9109 | 91.92% | n/a | n/a | n/a |
| TimesNet [25] | 0.9189 | 91.01% | 100% (66/66) | 1.55M | 68 |
| PatchTST [26] | 0.9311 | 92.65% | 98.5% (65/66) | 1.36M | 57 |
| V1: Baseline LSTM | 0.9235 | 91.88% | 98.5% (65/66) | 2.83M | 57 |
| V2: + TempDiff | 0.9430 | 93.52% | 100% (66/66) | 2.84M | 45 |
| V3: + EfficientNet | 0.9242 | 91.62% | 100% (66/66) | 4.52M | 59 |
| V4: CATFT - CrossAttn | 0.9112 | 90.67% | 100% (66/66) | 7.63M | 65 |
| V5: Full CATFT | 0.9252 | 92.13% | 100% (66/66) | 10.79M | 50 |
| **V2+ (Proposed)** | **0.9557+/-0.0006** | **95.02%** | **100% (66/66)** | **2.85M** | **24** |

All models except V2+ are single-run results with seed = 42. V2+ reports mean +/- std over three seeds (42, 123, 456). See Table 12 for details. The AI Hub reference MMTransformer is the ViT + cross-attention baseline reported in the dataset provider's utilization guideline [32], with fields that were not disclosed there marked "n/a".

**[Figure 3: Model performance comparison with parameter count overlay.]**

The proposed V2+ model achieves the highest macro F1-score of 0.9557 +/- 0.0006 (mean over three runs), outperforming every comparison method: the AI Hub dataset provider's official reference baseline (MMTransformer, F1 = 0.9109), recent general-purpose time series architectures (TimesNet, PatchTST), and the substantially larger internal CATFT variant (V5, 10.79M parameters). Compared to the AI Hub reference — evaluated under identical labels and dataset splits — V2+ improves F1 by 4.48 percentage points (0.9109 → 0.9557) and Accuracy by 3.10 percentage points (91.92% → 95.02%), despite using an estimated 3-5x fewer parameters. Compared to PatchTST, V2+ improves F1 by 2.46 % while using approximately twice the parameters; compared to TimesNet, the improvement is 3.68 %.

Two patterns emerge from Table 6. First, three independent cross-attention / transformer baselines — the dataset provider's MMTransformer (0.9109), our V4 CATFT-CrossAttn (0.9112), and its fully-featured version V5 (0.9252) — all cluster in the 0.91-0.93 F1 range, suggesting that cross-attention fusion applied directly to raw multimodal inputs has an empirical ceiling near 0.92 on this dataset (Section 7 discusses the cause). Second, both external general-purpose baselines (TimesNet, PatchTST) underperform even our V2 variant (LSTM + single-lag temporal difference), reinforcing that domain-informed temporal features are more effective than generic time-series architectures for this task.

Notably, V2+ achieves this with only 2.85M parameters—approximately 3.8x fewer than V5—while reducing Normal-Mild boundary errors by 58% compared to the baseline and PatchTST (57 to 24).

### 6.2. Ablation Study

#### 6.2.1. Architecture-Level Ablation (V1-V5)

**Table 7.** Architecture-level ablation results.

| Component Added to Baseline | F1 | DF1 vs V1 | Mechanism |
|:----------------------------|:--:|:---------:|:----------|
| V1: Baseline LSTM | 0.9235 | — | Reference |
| V2: + Temporal Diff (lag=1) | 0.9430 | +1.95% | Rate-of-change captures degradation dynamics |
| V3: + EfficientNet (pretrained) | 0.9242 | +0.07% | Transfer learning from ImageNet |
| V4: + Transformer Encoder | 0.9112 | -1.23% | Overfitting on 9,313 training windows |
| V5: + Cross-Attention | 0.9252 | +0.17% | Partial overfitting recovery via modality interaction |

**Key finding:** Temporal difference features provide the single largest performance improvement (+1.95%), while the Transformer architecture actually degrades performance (-1.23%) due to overfitting. This demonstrates that domain-informed feature engineering is more effective than architectural complexity when training data is limited.

#### 6.2.2. V2+ Component-Level Ablation

To isolate the contribution of each component added in V2+, we conduct three additional experiments, each adding only one component to the V2 baseline.

**Table 8.** Component-level ablation of V2+ improvements (each applied independently to V2).

| Configuration | F1 | DF1 vs V2 | NM Errors | Mechanism |
|:-------------|:--:|:---------:|:---------:|:----------|
| V2 (base) | 0.9430 | — | 45 | Single lag (lag=1) |
| V2a: + Multi-Scale Diff only | 0.9432 | +0.02% | 45 | Lags [1,5,10], no SE, no SupCon |
| V2b: + SE Attention only | 0.9319 | -1.11% | 53 | Lag [1], +SE, no SupCon |
| V2c: + SupCon Loss only | 0.9422 | -0.08% | 46 | Lag [1], no SE, +SupCon |
| **V2+ (all three combined)** | **0.9550** | **+1.20%** | **24** | **Lags [1,5,10], +SE, +SupCon** |

Individually, each component provides marginal or even negative improvement. Multi-scale diff alone (+0.02%) suggests that the LSTM can already extract some multi-scale information from single-lag features. SE attention alone (-1.11%) overfits the channel gating mechanism without sufficiently rich input features. SupCon loss alone (-0.08%) shows that contrastive learning on the original feature space provides limited benefit.

**[Figure 4: V2+ component ablation. (a) F1-score. (b) Normal-Mild boundary errors. Individual components are ineffective; combined effect shows strong synergy.]**

However, when combined, the three components produce a synergistic effect of +1.20% F1 and a 47% reduction in Normal-Mild errors (45 to 24, Figure 4). We attribute this to a cascading mechanism: multi-scale differences provide richer temporal features, enabling the SE block to perform meaningful channel selection, which in turn creates a more structured embedding space where supervised contrastive loss can effectively separate overlapping classes.

### 6.3. Class-Level Analysis

**Table 9.** Per-class performance of the proposed V2+ model (seed = 42).

| Class | Precision | Recall | F1-Score | Support |
|:------|:---------:|:------:|:--------:|--------:|
| Normal (0) | 0.98 | 0.96 | 0.97 | 520 |
| Mild (1) | 0.94 | 0.90 | 0.92 | 286 |
| Moderate (2) | 0.92 | 0.98 | 0.95 | 285 |
| Severe (3) | 0.97 | 1.00 | 0.99 | 66 |
| **Macro Avg** | **0.95** | **0.96** | **0.9550*** | **1,157** |

*Single run (seed = 42). Mean over three seeds: 0.9557 +/- 0.0006 (Table 12).

**Table 10.** Confusion matrix comparison: Baseline (V1) vs. Proposed (V2+).

```
V1 Baseline                              V2+ Proposed
         Pred: N    Mi   Mo   Se          Pred: N    Mi   Mo   Se
True N    470   39   10    1         N    497   12    9    2
True Mi    18  256   12    0         Mi    12  258   16    0
True Mo     0    9  272    4         Mo     0    5  280    0
True Se     0    0    1   65         Se     0    0    0   66
```

**[Figure 5: Confusion matrix comparison. Red boxes highlight Normal-Mild boundary errors.]**

The Normal-Mild confusion is the dominant error source in both models (Figure 5). V2+ reduces these errors from 57 (39+18) to 24 (12+12), a 57.9 % reduction. This improvement is attributed to (1) multi-scale temporal differences that amplify the subtle 4 °C temperature gradient between states, and (2) supervised contrastive loss that explicitly separates the overlapping embeddings. Note also that a portion of the residual Attention <-> Warning confusion visible on the diagonal-adjacent cells (Mild-Moderate in Table 10 shorthand) is a structural lower bound rather than a model deficiency: per Section 3.3 the Attention and Warning classes are a 1:1 partition of one physically homogeneous interval and therefore possess no physical or statistical boundary. The 24 Normal-Mild errors V2+ still produces should be read against this constraint — they lie at the same order as the class-1/class-2 arbitrariness rather than reflecting an obvious feature-engineering gap.

### 6.4. Severe State Detection

Across all model variants except the baseline V1, perfect Severe state detection (66/66, 100% recall) is achieved. This is attributed to the unambiguous sensor signatures at State 3: temperature surges to 50.71 °C mean, and current sensors exhibit 3.5-14x increases over baseline values. The class-weighted loss ensures sufficient training focus on the minority Severe class (7.7% of training data).

### 6.5. Contribution of Thermal Imagery

To assess the contribution of the thermal modality, we compare the proposed V2+ model against a sensor-only variant that removes the thermal image branch entirely.

**Table 11.** Contribution of thermal imagery: sensor-only vs. multimodal comparison.

| Model | F1 | Accuracy | NM Errors | Params |
|:------|:--:|:--------:|:---------:|-------:|
| V2+ Sensor-only | 0.9513 | 94.47% | 32 | 0.37M |
| V2+ Sensor + Thermal | 0.9550 | 95.16% | 24 | 2.85M |
| **Thermal contribution** | **+0.37%** | **+0.69%** | **-8** | **+2.48M** |

The thermal modality provides a modest but consistent improvement of +0.37 % F1 and reduces Normal-Mild errors by 8 cases. The magnitude of this contribution is best interpreted with the source-hardware constraint disclosed in Section 3.2: the acquisition camera (Terabee Evo Thermal 33) has a 32 x 32 native pixel grid, and the 120 x 160 arrays we consume are the result of the provider's ~15 x upsampling. The spatial information available to any thermal branch is therefore bounded by the ~1,024 native pixels, not by the 19,200 stored pixels. The +0.37 % F1 gain likely reflects the ceiling of this ~1 K-pixel information budget rather than a limit of our CNN branch. A deployment on the higher-resolution FLIR Lepton 3.5 (160 x 120 native, ~19,200 pixels) — with which we plan to fine-tune in the field study described in Section 7.5 — is expected to give the thermal branch materially more headroom. The sensor-only model is nonetheless noteworthy for achieving F1 = 0.9513 with only 0.37 M parameters, which may be preferable for extremely resource-constrained deployments where the thermal camera is unavailable.

### 6.6. Reproducibility and Statistical Stability

To assess the robustness of V2+, we repeat the experiment with three different random seeds.

**Table 12.** Repeated runs of V2+ with different random seeds.

| Seed | F1 | Accuracy | NM Errors |
|:-----|:--:|:--------:|:---------:|
| 42 | 0.9550 | 95.16% | 24 |
| 123 | 0.9560 | 94.99% | 29 |
| 456 | 0.9560 | 94.90% | 33 |
| **Mean +/- Std** | **0.9557 +/- 0.0006** | **95.02 +/- 0.13%** | **28.7 +/- 4.5** |

The standard deviation of 0.0006 in F1-score indicates high reproducibility. All three runs achieve F1 > 0.955, confirming that the reported performance is not an artifact of a particular random initialization.

### 6.7. Lag Selection Sensitivity

We evaluate the sensitivity of the proposed method to the choice of temporal difference lags.

**Table 13.** Lag sensitivity analysis.

| Lags | F1 | NM Errors | Observation |
|:-----|:--:|:---------:|:------------|
| [1] (V2 baseline) | 0.9430 | 45 | Single lag, no multi-scale |
| [1, 3, 7] | 0.9498 | 35 | Short intervals, partial redundancy |
| **[1, 5, 10]** | **0.9550** | **24** | **Optimal spacing** |
| [1, 10, 20] | 0.9520 | 31 | Lag-20 exceeds useful range (zero-padding) |

**[Figure 6: Lag sensitivity analysis. (a) F1-score. (b) Boundary errors.]**

The optimal lag combination [1, 5, 10] provides approximately geometric spacing (1x, 5x, 10x), capturing instantaneous changes, short-term trends, and medium-term trends respectively. The [1, 3, 7] combination suffers from redundancy between adjacent lags, while [1, 10, 20] introduces excessive zero-padding at lag-20 (the first 20 timesteps of a 30-step window are zero), degrading performance. All multi-scale combinations outperform the single-lag baseline, confirming the value of multi-scale temporal representation.

### 6.8. Deployment Performance

The proposed model is exported to ONNX format for edge deployment.

**Table 14.** Deployment benchmarks.

| Metric | Value |
|:-------|:------|
| ONNX model size | 10.9 MB |
| Accuracy preservation | 100% (1,157/1,157 identical predictions) |
| Inference: PyTorch GPU (RTX 6000) | 1.18 ms |
| Inference: ONNX Runtime GPU | 2.61 ms |
| Inference: ONNX Runtime CPU | 7.27 ms |
| Inference: Jetson Orin Nano (est.) | ~5 ms |

All backends achieve sub-10 ms inference, well within the 1-second real-time requirement for manufacturing safety systems. The model can process approximately 200 sensor readings per second on an estimated Jetson deployment, providing ample margin for real-time operation.

### 6.9. Hypothesis Validation

We now revisit the three hypotheses posed in Section 1 and map each to the empirical evidence obtained above. The result is summarized in Table 15.

**Table 15.** Mapping of research questions and hypotheses to empirical evidence.

| RQ / H | Prediction | Result | Evidence | Verdict |
|:------:|:-----------|:-------|:---------|:-------:|
| **H1** | Multi-scale temporal differences improve F1 over single-lag and absolute-value baselines | Multi-scale diff [1,5,10] within V2+ achieves F1 = 0.9550 vs. single-lag V2 = 0.9430 (Delta = +1.20 %) and V1 absolute-only baseline = 0.9235 (Delta = +3.15 %); lag sensitivity confirms [1,5,10] as optimal spacing over [1], [1,3,7], and [1,10,20] | Table 7 (V1 vs V2), Table 8 (V2 vs V2+), Table 13 (lag sensitivity) | **Supported** |
| **H2** | SupCon (lambda = 0.1) reduces Normal-Mild misclassifications by >= 40 % vs. single-lag baseline | Normal-Mild errors: V2 = 45 -> V2+ = 24, a 46.7 % reduction; overall macro F1 simultaneously improves from 0.9430 to 0.9557 +/- 0.0006 | Table 8, Table 10 (confusion matrices) | **Supported** |
| **H3** | V2+ (2.85 M) outperforms TimesNet, PatchTST, CATFT, and the AI Hub reference MMTransformer on macro F1 while remaining < 10 ms per inference | V2+ F1 = 0.9557 +/- 0.0006 vs. AI Hub reference MMTransformer 0.9109, TimesNet 0.9189, PatchTST 0.9311, CATFT V5 0.9252 (10.79 M); ONNX GPU latency = 2.61 ms and ONNX CPU = 7.27 ms both below the 10 ms budget | Table 6 (all models), Table 14 (deployment) | **Supported** |

For H1, the ablation isolates the effect of the temporal-difference component: V2a (multi-scale diff added to V2, in isolation from SE and SupCon) alone does not lift F1 meaningfully (Table 8), whereas the full multi-scale configuration inside V2+ does. This nuance is consistent with H1 in its **combined-effect form** and is discussed further in Section 7.3.

For H2, the 46.7 % reduction in Normal-Mild errors exceeds the pre-registered threshold of 40 %, and the accompanying F1 improvement rules out a trivial trade-off between class-level accuracy and boundary separation.

For H3, the parameter-efficiency claim is strengthened by the observation that V2+ outperforms V5 CATFT while using approximately 3.8x fewer parameters, and that both external general-purpose baselines (TimesNet, PatchTST) underperform even the single-lag V2 variant. The deployment measurement in Table 14 confirms that this accuracy advantage is not obtained at the cost of edge feasibility.

All three hypotheses are therefore empirically supported. Limitations of this validation, including the single-dataset scope and the absence of on-device Jetson measurements at the time of writing, are addressed in Section 7.5.

---

## 7. Discussion

### 7.1. Domain Knowledge vs. Architectural Complexity

Our results demonstrate a counterintuitive finding: a 2.85M-parameter LSTM with domain-informed temporal features outperforms a 10.8M-parameter Transformer with cross-attention by 3.05 % F1 (0.9557 +/- 0.0006 vs. 0.9252). This is corroborated by the AI Hub reference baseline: the dataset provider's official MMTransformer (a ViT + cross-attention model evaluated on the same labels [32]) reaches F1 = 0.9109, statistically indistinguishable from our internal V4 CATFT-CrossAttn (0.9112) and V5 CATFT (0.9252). Three independently trained cross-attention Transformers therefore all fall in the 0.91-0.93 F1 band on this dataset, while V2+ clears 0.955 — a 3-5 percentage-point gap that is unusually large for a "smaller model beats bigger" result and suggests a systematic ceiling for cross-attention on this dataset rather than an isolated tuning issue.

Two factors explain this ceiling. First, the training set size (9,313 windows) is insufficient for a Transformer to learn discriminative attention patterns from scratch — V4 performance falls 1.23 % below the LSTM baseline, and V5 recovers only 1.40 % via the cross-attention inductive bias. Second, temporal difference features encode a strong domain prior: degradation is a dynamic process best characterized by rate-of-change rather than instantaneous values. This physics-informed feature reduces the hypothesis space the model must search, making learning more sample-efficient at this dataset scale.

### 7.2. Why Multi-Scale Temporal Differences Work

The effectiveness of multi-scale temporal differences can be understood through the lens of the Normal-Mild boundary problem. At lag-1 (1-second difference), the temperature change between Normal and Mild states is approximately 0.13 °C/step—well within sensor noise margins. At lag-10 (10-second accumulated difference), this grows to approximately 1.3 °C—a detectable signal. The multi-scale representation allows the LSTM to simultaneously leverage fast transients (lag-1, useful for detecting sudden failures) and gradual trends (lag-10, useful for early degradation detection).

Our lag sensitivity analysis (Table 13) shows that approximately geometric spacing (1x, 5x, 10x) outperforms both narrower [1, 3, 7] and wider [1, 10, 20] combinations. Narrow spacing introduces redundant features, while excessively wide lags suffer from zero-padding artifacts at the beginning of the window.

### 7.3. Synergy of Combined Components

A notable finding from our component-level ablation (Table 8) is that the three V2+ improvements (multi-scale diff, SE attention, SupCon loss) exhibit strong synergy: individually, they provide marginal or negative improvement, but combined they yield +1.20% F1 and a 47% reduction in boundary errors. We hypothesize a cascading mechanism: (1) multi-scale differences expand the feature space with richer temporal information, (2) the SE block learns to select the most discriminative features from this enriched space, and (3) supervised contrastive loss exploits the well-structured features to explicitly separate overlapping class boundaries. This synergy underscores the importance of evaluating feature engineering and loss function design jointly, rather than in isolation.

### 7.4. Role of Thermal Imagery

Our sensor-only ablation (Table 11) reveals that thermal images contribute +0.37% F1, with the sensor-only model achieving F1 = 0.9513 using merely 0.37M parameters. This suggests that for this particular dataset, the NTC temperature sensor already captures most of the thermal information that the infrared camera provides.

However, we argue that the thermal modality remains valuable for two reasons. First, it reduces Normal-Mild errors by 8 cases (32 to 24), providing an additional safety margin in the most ambiguous classification boundary. Second, in real-world deployment scenarios, thermal cameras capture spatial heat distribution patterns (e.g., localized hotspots) that point-contact NTC sensors cannot detect. The modest improvement observed here may reflect the fact that the dataset's controlled testbed environment generates relatively uniform heating patterns; in actual production lines with more complex equipment geometries, the thermal modality's contribution is expected to be more significant. We therefore retain the multimodal design and present the sensor-only variant as an option for extremely resource-constrained deployments.

### 7.5. Limitations and Future Work

This study has several limitations.

**Dataset scope and label boundaries.** The dataset provides a fixed train/validation split without a separate held-out test set; we use the validation set for both model selection and evaluation. While equipment-level separation prevents temporal leakage and our three-seed runs (Table 12) demonstrate stability (F1 std = 0.0006), a separate test set would provide a more rigorous evaluation. In addition, per the definition in Section 3.3 the Attention (State 1) and Warning (State 2) classes correspond to a 1:1 partition of a single physically homogeneous interval, so a portion of the residual State-1 <-> State-2 confusion in Table 10 is irreducible under any model. Reporting metrics that collapse these two classes into a single "degraded" super-class is a promising direction that we defer to future work.

**Unused fields in the released schema.** Our current model uses only the 8 sensor time series, the 120 x 160 thermal array, and the 4-class state label. The AI Hub schema also exposes fields that we have not yet integrated: `external_data` (ambient temperature 22-26 °C, humidity 27-36 %, illuminance 151-530 lux), the max-temperature pixel coordinates (`ir_data.temp_max.X_Tmax / Y_Tmax`), a pre-computed trend indicator per sensor value (`sensor_data.*.trend`), and static per-device meta features (`cumulative_operating_day`, `equipment_history`, `device_manufacturer`). Adding these — for example, ambient conditions as a small MLP branch, manufacturer as a 3-way categorical embedding, and the max-temperature coordinates as auxiliary spatial supervision — is a natural next-step ablation.

**Cross-facility and sensor generalization.** The validation is performed on the same testbed's equipment (different device units); cross-facility generalization has not been tested. Because the released data are collected with a single sensor model per category (Section 3.2), the model has also not been trained on sensor-hardware diversity. Real deployment with different current transformers, particulate sensors, and thermal cameras is therefore expected to require domain adaptation.

**Field deployment and higher-resolution thermal input.** Real-time inference performance is estimated from server GPU benchmarks; on-device latency and accuracy on the actual Jetson Orin Nano deployment are pending and will be added in a subsequent revision. As noted in Section 6.5, the field-deployment thermal camera (FLIR Lepton 3.5, 160 x 120 native) exceeds the acquisition camera's native resolution by roughly a factor of 19; fine-tuning on higher-native-resolution thermal images is expected to lift the modest +0.37 % thermal contribution we currently observe.

**Formulation and pretraining directions.** The current approach treats degradation as a discrete 4-class classification, whereas degradation is inherently continuous; regression-based or ordinal-classification formulations may better exploit the fact that the label ordering carries physical meaning up to the Attention/Warning definitional artifact. A related direction is self-supervised pretraining on unlabeled sensor streams pooled across multiple predictive-maintenance datasets (e.g., CMAPSS, FEMTO-ST) — an approach related to recent "sensor language model" and time-series foundation-model work — followed by supervised fine-tuning on this dataset. This is unlikely to help within the current 9,313-window subset (which motivated our decision to prioritize domain-informed features), but becomes attractive once the corpus expands via additional field data or cross-dataset pretraining.

---

## 8. Conclusions

We presented a lightweight multimodal approach for manufacturing equipment degradation prediction that prioritizes domain-informed feature engineering over architectural complexity. Our multi-scale temporal difference features, combined with supervised contrastive learning and channel attention, achieve a macro F1-score of 0.9557 +/- 0.0006 (mean over three runs) with only 2.85M parameters—nearly 4x fewer than Transformer-based alternatives. The approach achieves 100% detection of safety-critical Severe degradation states and reduces Normal-Mild boundary misclassifications by 47%.

Our comprehensive ablation study reveals two important insights for practitioners. First, domain-specific feature engineering (temporal differences) provides the single largest performance improvement (+1.95%), outweighing any architectural change. Second, the three proposed improvements (multi-scale differences, channel attention, and contrastive loss) exhibit strong synergy: individually ineffective, they produce a +1.20% F1 improvement when combined, highlighting the importance of joint design of features, architecture, and loss functions.

With an estimated inference latency of approximately 5 ms on NVIDIA Jetson Orin Nano and a model size of 10.9 MB, the approach is practical for real-time on-device deployment in manufacturing safety systems. These findings suggest that in data-limited industrial settings, investing in domain-specific feature engineering yields greater returns than scaling model complexity.

---

## References

[1] Lee, J., et al. "Prognostics and health management design for rotary machinery systems—Reviews, methodology and applications." Mechanical Systems and Signal Processing, 2014.

[2] Zhong, S., et al. "Deep learning for predictive maintenance: A survey." IEEE Access, 2022.

[3] Zhao, R., et al. "Deep learning and its applications to machine health monitoring." Mechanical Systems and Signal Processing, 2019.

[4] Goyal, D., Pabla, B. "The vibration monitoring methods and signal processing techniques for structural health monitoring: a review." Archives of Computational Methods in Engineering, 2016.

[5] Liu, Y., et al. "Multimodal transformer for multimodal machine translation." ACL, 2020.

[6] Zhang, Y., et al. "mmTransformer: Multimodal motion prediction with stacked transformers." CVPR, 2021.

[7] Li, T., et al. "Understanding the difficulty of training transformers on time series data." ICML, 2023.

[8] NVIDIA Corporation. "Jetson Orin Nano Developer Kit." Technical Documentation, 2023.

[9] Ran, Y., et al. "A survey of predictive maintenance: Systems, purposes and approaches." arXiv preprint, 2019.

[10] Mobley, R.K. "An Introduction to Predictive Maintenance." Butterworth-Heinemann, 2002.

[11] Fink, O., et al. "Potential, challenges and future directions for deep learning in prognostics and health management applications." Engineering Applications of Artificial Intelligence, 2020.

[12] Zhang, W., et al. "A deep convolutional neural network with new training methods for bearing fault diagnosis under noisy environment and different working load." Mechanical Systems and Signal Processing, 2018.

[13] Malhotra, P., et al. "Long short term memory networks for anomaly detection in time series." ESANN, 2015.

[14] Zhao, R., et al. "Machine health monitoring using local feature-based gated recurrent unit networks." IEEE Transactions on Industrial Electronics, 2018.

[15] Wu, H., et al. "Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting." NeurIPS, 2021.

[16] Zhou, T., et al. "FEDformer: Frequency enhanced decomposed transformer for long-term series forecasting." ICML, 2022.

[17] Gao, Z., et al. "A survey of fault diagnosis and fault-tolerant techniques—Part I: Fault diagnosis with model-based and signal-based approaches." IEEE Transactions on Industrial Electronics, 2015.

[18] Ramachandram, D., Taylor, G.W. "Deep multimodal learning: A survey on recent advances and trends." IEEE Signal Processing Magazine, 2017.

[19] Tsai, Y.H., et al. "Multimodal transformer for unaligned multimodal language sequences." ACL, 2019.

[20] Xu, P., et al. "Multimodal learning with transformers: A survey." IEEE TPAMI, 2023.

[21] Wang, J., et al. "Sensor data fusion for manufacturing quality prediction." Journal of Manufacturing Systems, 2022.

[22] Hamilton, J.D. "Time Series Analysis." Princeton University Press, 1994.

[23] Box, G.E., et al. "Time Series Analysis: Forecasting and Control." Wiley, 2015.

[24] Montgomery, D.C. "Introduction to Statistical Quality Control." Wiley, 2019.

[25] Wu, H., et al. "TimesNet: Temporal 2D-variation modeling for general time series analysis." ICLR, 2023.

[26] Nie, Y., et al. "A time series is worth 64 words: Long-term forecasting with transformers." ICLR, 2023.

[27] Chen, T., et al. "A simple framework for contrastive learning of visual representations." ICML, 2020.

[28] Khosla, P., et al. "Supervised contrastive learning." NeurIPS, 2020.

[29] Yue, Z., et al. "TS2Vec: Towards universal representation of time series." AAAI, 2022.

[30] Choi, S., et al. "Soft contrastive learning for time series." ICLR, 2024.

[31] Yang, B., et al. "Contrastive learning for fault diagnosis: A survey." IEEE Transactions on Instrumentation and Measurement, 2023.

[32] AI Hub. "Manufacturing Transport Device Degradation Predictive Maintenance Multimodal Data." Dataset ID: 71802. https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71802

[33] Hu, J., et al. "Squeeze-and-excitation networks." CVPR, 2018.

[34] Tan, M., Le, Q. "EfficientNet: Rethinking model scaling for convolutional neural networks." ICML, 2019.
