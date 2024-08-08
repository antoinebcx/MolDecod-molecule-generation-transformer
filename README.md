# MolDecod: Decoder-only transformer for molecule generation

MolDecod is a 5M-parameter decoder-only transformer using rotary positional encoding, for SMILES molecule generation.

It was trained on ~2.7M molecules (~90M tokens), from the high-quality MOSES and ChEMBL datasets [[+]](https://tdcommons.ai/generation_tasks/molgen/), and achieves an impressive performance for its size.

___

## Repository

This repository contains:
- MolDecod and its tokenizer in `/models`
- A `streamlit.py` app to interact with the model
- Utils functions in `/utils`
- Notebooks to train, evaluate and use the model in `/notebooks`

___

## Model characteristics

MolDecod is a decoder-only transformer (GPT-like), using rotary positional encoding. It has a model dimension of 256, 4 attention heads and 4 encoding layers, resulting with the buffers in a total of 5 million parameters.

On 10,000 generated molecules (with temperature 0.7), it obtains the following metrics:
- Validity: 0.95
- Uniqueness: 0.95
- Diversity: 0.87
- Novelty: 0.93

___

## App

You can launch a streamlit app to interact with the model. Download the repo, open a terminal window, install the requirements and run the following command:
```
streamlit run streamlit.py
```

You can generate molecules as well as visualize their structure and molecular properties.

&nbsp;

<img width="742" alt="image" src="https://github.com/user-attachments/assets/8b6fec5b-fef4-4475-8358-8475dee558f1">

<img width="742" alt="image" src="https://github.com/user-attachments/assets/bf5f396f-7618-4d56-aac1-56f444a8ef9a">
