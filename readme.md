# High Innovation! MSD-BiGRUA Attention: An innovative predictive model based on multi-scale decomposition and GlobalAttention optimization for BiGRU.

1. Introduction
This period proposes a high-innovation model for wind power prediction: MSD-BiGRUAttention, which is based on multi-scale decomposition and BiGRU global attention mechanism for wind power prediction. The model takes multi-variable wind power time series as input, firstly divides the original sequence into temporal patterns of different time granularities through the multi-scale decomposition module; subsequently, independent BiGRU-GlobalAttention branches are constructed for each scale to extract bidirectional sequential dynamic features at different scales; finally, the contribution weights of different scale branches are adaptively assigned through the scale attention fusion module, and the future wind power prediction results are output.
The core idea of the model can be summarized as follows:
Multi-scale decomposition → Parallel BiGRU-Attention modeling → Adaptive scale fusion → Wind power prediction
This model not only captures short-term fluctuations, medium-term changes, and long-term trends in wind power sequences but also highlights key historical time steps through the GlobalAttention mechanism, thereby enhancing the model's ability to perceive the temporal features of multi-feature sequences. (Innovation point!)
We also provide detailed materials, explanation documents, and video tutorials, including how to replace your own dataset, parameter adjustment tutorials, etc. The code is commented line by line with detailed parameter introductions:
● Datasets: Wind power dataset from a wind farm, electricity dataset, wind speed dataset, etc.
● Environment framework: Python 3.11, PyTorch 2.1 and above versions can run
● Model scores: Test set 0.98
● Use cases: Paper requirements, graduation design requirements
● Code guarantee: Code comments are detailed, ready to run without modification.
1. Innovation Model Introduction
1.1 Overall Process of the Model
The overall process of MSD-BiGRUAttention is as follows:
(1) Input multivariate time series:
X ∈ R^{B×L×C}
where:
B represents the batch size
L represents the length of the historical window
C represents the number of input feature variables
(2) The model first performs multi-scale decomposition:
Scale 1: Raw fine-scale sequence
Scale 2: Medium-scale downsampled sequence
Scale 3: Coarse-scale downsampled sequence
Subsequently, each scale enters an independent BiGRU-GlobalAttention branch:
Scale 1 → BiGRU-Attention
Scale 2 → BiGRU-Attention
Scale 3 → BiGRU-Attention
(3) Each branch outputs a scale-level feature representation, and then calculates the importance weights of different scales through ScaleAttentionFusion, followed by adaptive fusion. Finally, the fused features are input into a linear prediction head to output future wind power.

1.2 Core module one: MultiScaleDecomposition - MultiScale Decomposition Module
MultiScaleDecomposition is the first step of the model, mainly used for segmenting the original multivariate time series into representations at different temporal scales.
This module performs average pooling downsampling along the temporal dimension on the input sequence, obtaining temporal patterns of different time granularities.
For example, if the input sequence length is 24, with 3 scales and downsampling rate of 2, it will be obtained as:
Scale 1: [B, 24, C]
Scale 2: [B, 12, C]
Scale 3: [B, 6, C]
This means that the model retains both the original fine-grained sequences, medium-scale sequences, and coarse-grained trend sequences.

1.3 Core module two: Multi-scale parallel BiGRU-GlobalAttention branch
The lengths and change characteristics of different scale sequences are different. Fine-scale sequences preserve more local details, suitable for capturing short-term fluctuations; coarse-scale sequences become smoother after downsampling, suitable for capturing long-term trends. If all scales are fused prematurely, it may result in mixed scale information. Therefore, this model establishes independent BiGRU branches for each scale to allow each branch to learn the temporal dynamics under its scale. GlobalAttention is a mechanism used to enhance the model's focus on different parts of the input sequence. In the BiGRU model, the global attention mechanism helps the model better concentrate on the most relevant parts of the input sequence, thereby improving the model's performance and generalization ability. At each time step, the global attention mechanism calculates a weight vector indicating the model's focus on different parts of the input sequence. These weights are then applied to the feature representations of the BiGRU outputs, allowing the model to focus more specifically on important temporal features by weighting all locations, enhancing the model's perception capability of multi-feature sequence temporal features.

1.4 Core module three: ScaleAttentionFusion Scale adaptive fusion
ScaleAttentionFusion will calculate an importance weight for each scale branch:
γ1, γ2, γ3
Then, the representations at each scale are weighted for fusion:
z = γ1z1 + γ2z2 + γ3z3
Where:
γ1 + γ2 + γ3 = 1
The mechanism enables the model to adaptively judge the importance of different scales based on the current sample state. In other words, the "adaptive" aspect here mainly manifests in two levels:
1. GlobalAttention adaptively selects key time steps within each scale;
2. ScaleAttentionFusion adaptively chooses a more important time granularity between scales.
This allows the model to simultaneously possess intra-scale attention and inter-scale attention.
2 Data Set Introduction for Electric Power Transformers
2.1 Importing Data
The Wind Power Power Data set consists of 35040 samples with 15 features, visualizing the first 6000 samples:

2.2 Data Set Preparation and Preprocessing
For detailed instructions, please refer to the provided documentation!
3 High-Innovative Prediction Model Based on MSD-BiGRUAttention
3.1 Definition of the MSD-BiGRUAttention prediction network model
3.2 Setting parameters and training the model
Trained for 50 epochs, with an MSE of 0.001239, the MSD-BiGRUAttention prediction effect is significant. This model extracts multivariate temporal patterns at different time scales through MultiScaleDecomposition, captures bidirectional temporal dynamics at each scale through BiGRU, enhances the contribution of key historical time steps through GlobalAttention, and adaptively fuses information from different scales through ScaleAttentionFusion. Overall, this model has the following advantages:
● It can capture short-term disturbances, medium-term changes, and long-term trends
● It can enhance the temporal feature perception ability of multi-feature sequences
● It can adaptively judge the importance of different scales
● Clear structure, controllable parameters, easy to train
● Visualization of attention weights with strong explanatory power
4 Results Visualization and Model Evaluation
\t4.1 Prediction results visualization
\t4.2 Model evaluation
MSD-BiGRUAttention is not only suitable for wind power power prediction but can also be extended to photovoltaic power prediction, power load prediction, equipment status prediction, transformer oil temperature prediction, and other multivariate time series prediction tasks.
Previously interesting content:
Informer prediction model collection: New feature importance analysis!
Fast Fourier Transform brute force surge! Highly innovative time series classification model based on time-frequency feature fusion
!STL-VMD secondary decomposition + Informer-LSTM parallel prediction model
Continued updates! Attention mechanism + Lithium Battery Remaining Life Prediction Collection
Direct writing in one area! CEEMDAN decomposition + Informer-LSTM +XGBoost combination prediction model
Based on Informer-SENet, a comparison of photovoltaic power generation prediction models! Six sets of comparative prediction models, directly written for thesis or small thesis!
!Based on ICEEMDAN+SHAP explainable analysis of Lithium Battery Remaining Life Prediction high innovation model!
!Based on Informer-BiGRUGATT-CrossAttention prediction model
High innovation! Based on ICEEMDAN+MSCNN-BiGRU-Attention parallel prediction model
Air Quality PM2.5 concentration high innovation prediction model! Based on VMD rolling decomposition + attention mechanism parallel prediction model
END
\t2. Friends who like this can click to follow, give the article a "like", share on WeChat groups or discuss groups

## Images

<!-- PAYMENT_BLOCK -->

## Get this dataset

- Payment link (Stripe): [Buy now](https://buy.stripe.com/3cs8yP7sY87d0vu9AB)
- Payment QR code: ![Payment QR code](pay.png)

<!-- /PAYMENT_BLOCK -->
