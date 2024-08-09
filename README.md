# MolDecod: Transformer for molecule generation

MolDecod is a 5M-parameter decoder-only transformer using rotary positional encoding, for SMILES molecule generation.

It was trained on ~2.7M molecules (~90M tokens), from the high-quality MOSES and ChEMBL datasets [[+]](https://tdcommons.ai/generation_tasks/molgen/), and achieves an impressive performance for its size.

## Repository

This repository contains:
- MolDecod and its tokenizer in `/models`
- A `streamlit.py` app to interact with the model
- Utils functions in `/utils`
- Notebooks to train, evaluate and use the model in `/notebooks`
- `TechnicalReport.md`, for technical details

## Model characteristics

MolDecod is a decoder-only transformer model (GPT-like) using rotary positional encoding, with a model dimension of 256, 4 attention heads, and 4 decoder layers. This results in ~5 million parameters, with ~4 million of them being trainable.

On 10,000 generated molecule, it obtains the following metrics:
| Temperature | Validity | Uniqueness | Diversity | Novelty | KL Divergence | Fragment Similarity | Scaffold Diversity |
|-------------|----------|------------|-----------|---------|---------------|---------------------|--------------------|
| 0.1         | 1.00     | 0.04       | 0.76      | 0.9455  | 6.4742        | 0.0545              | 0.0148             |
| 0.25        | 1.00     | 0.49       | 0.81      | 0.8347  | 4.3664        | 0.1653              | 0.1398             |
| 0.5         | 0.98     | 0.95       | 0.85      | 0.8768  | 5.7033        | 0.1237              | 0.4556             |
| 0.7         | 0.96     | 0.95       | 0.87      | 0.9240  | 5.6936        | 0.0778              | 0.6540             |
| 0.9         | 0.88     | 0.88       | 0.88      | 0.9562  | 5.3179        | 0.0502              | 0.7524             |

See `TechnicalReport.md` for more details and explanations.

## App

You can launch a streamlit app to interact with the model. Download the repo, open a terminal window, install the requirements and run the following command:
```
streamlit run streamlit.py
```

You can generate molecules as well as visualize their structure and molecular properties.

&nbsp;

<img width="742" alt="image" src="https://github.com/user-attachments/assets/8b6fec5b-fef4-4475-8358-8475dee558f1">

<img width="742" alt="image" src="https://github.com/user-attachments/assets/bf5f396f-7618-4d56-aac1-56f444a8ef9a">
