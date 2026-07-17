# Drug-Disease Interaction (DDSI) Literature Screening Classifier

Code and data for the manuscript *"Development of a machine-learning algorithm to
predict the relevance of scientific articles for drug-disease interactions"*
(de Ruiter, Habets, et al.).

This repository contains the code and data used to fine-tune and evaluate two
PubMedBERT-based classifiers for automating relevance screening of PubMed
articles for drug–disease interaction (DDSI) assessment reports.

## Models

Two fine-tuned model checkpoints are hosted on the Hugging Face Hub (too large
for this repository):

| Model | Input | Description | Weights |
|---|---|---|---|
| **Model 1** | Title + abstract | Fine-tuned for 2 epochs | [PH1230/ddsi-model-v1](https://huggingface.co/PH1230/ddsi-model-v1) |
| **Model 2** | DDSI + intervention description + title + abstract | Fine-tuned for 1 epoch | [PH1230/ddsi-model-v2](https://huggingface.co/PH1230/ddsi-model-v2) |

Both models are fine-tuned from [`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract).

Load either model directly with `transformers`:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("PH1230/ddsi-model-v1")
model = AutoModelForSequenceClassification.from_pretrained("PH1230/ddsi-model-v1")
```

## Repository contents

- Training and fine-tuning scripts
- Evaluation scripts (balanced test set and real-world-prevalence evaluation)
- Data splits (train / validation / test), including report-level grouping
- Scripts used to generate the figures and tables reported in the manuscript
- Original DDSI report data from which the results presented in the manuscript can be reproduced using the provided scripts.



