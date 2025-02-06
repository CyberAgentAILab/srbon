
# Stochastic Regularized Best-of-N Sampling (SRBoN)

This repository contains the implementation of **Stochastic Regularized Best-of-N Sampling (SRBoN)**.

The code is tested using Python 3.8 and CUDA 11.0
## Setup Instructions

### Step 1: Environment Setup

Create a virtual environment, then install the required dependencies:

```bash
# Create virtual environment
python3 -m venv env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Sample Collection and Metric Computation

#### Collect Samples

To collect samples from the model, use the following command. You can specify the dataset, model, and number of samples:

```bash
bash ./experiments/sample.sh -d [DATASETS] -m [MODEL] -s [NUMBER_OF_SAMPLES]
```

#### Compute Metrics

You can compute various utility metrics such as log probability, Wasserstein distance, and token length using the following scripts:

```bash
# Compute log probability
bash ./experiments/compute_logprob.sh -d [DATASETS] -m [MODEL] -s [NUMBER_OF_SAMPLES]

# Compute Wasserstein distance
bash ./experiments/compute_wd.sh -d [DATASETS] -m [MODEL] -s [NUMBER_OF_SAMPLES]

# Compute token length
bash ./experiments/compute_length.sh -d [DATASETS] -m [MODEL] -s [NUMBER_OF_SAMPLES]
```

#### Compute Reward Values

To compute reward values for specific datasets, you can use the following command. Here, specify the dataset, number of samples, and the reward type:

```bash
bash ./experiments/compute_reward.sh -d [DATASETS] -s [NUMBER_OF_SAMPLES] -i [REWARD_TYPE]
```

### Step 3: Running SRBoN

Finally, to compute the SRBoN values, run the following script:

```bash
python3 stochastic_rbon/stochastic_rbon.py --dataset [DATASETS] --ncandidates [NUMBER_OF_SAMPLES]
```

## Examples

Below is an example of running the SRBoN pipeline using the `alpaca` dataset, the `HuggingFaceH4/mistral-7b-sft-beta` model, and 100 samples.

```bash
# Collect 100 samples from the model
bash ./experiments/sample.sh -d alpaca -m HuggingFaceH4/mistral-7b-sft-beta -s 100

# Compute log probabilities for the collected samples
bash ./experiments/compute_logprob.sh -d alpaca -m HuggingFaceH4/mistral-7b-sft-beta -s 100

# Compute Wasserstein distance for the samples
bash ./experiments/compute_wd.sh -d alpaca -m HuggingFaceH4/mistral-7b-sft-beta -s 100

# Compute the token length of the samples
bash ./experiments/compute_length.sh -d alpaca -m HuggingFaceH4/mistral-7b-sft-beta -s 100
```

For computing reward values using different reward models:

```bash
# Compute reward values using the OpenAssistant reward model
bash ./experiments/compute_reward.sh -d alpaca -s 100 -i OpenAssistant/reward-model-deberta-v3-large-v2

# Compute reward values using the openbmb reward model
bash ./experiments/compute_reward.sh -d alpaca -s 100 -i openbmb/Eurus-RM-7b
```

Finally, running the **SRBoN** computation with 100 candidates:

```bash
python3 stochastic_rbon/stochastic_rbon.py --dataset alpaca --ncandidates 100
```
<!-- ## Reference
Yuki, I., Jinnai, Y., Morimura, T., Abe, K., Ariu, K., Sakamoto, M., and Uchibe, E. "Evaluation of Best-of-N Sampling Strategies for Language Model Alignment."

Bibtex:
```bash
@article{
anonymous2024evaluation,
title={Evaluation of Best-of-N Sampling Strategies for Language Model Alignment},
author={Anonymous},
journal={Submitted to Transactions on Machine Learning Research},
year={2024},
url={https://openreview.net/forum?id=H4S4ETc8c9},
note={Under review}
}
``` -->