# MolDecod: Decoder-only transformer for molecule generation

This repository contains:
- MolDecod, a small transformer model for SMILES molecule generation
- its tokenizer, a SentencePiece model
- utils functions and a streamlit app to interact with the model
- notebooks to train and use the model

___

## Model

MolDecod is a 5M-parameter decoder-only transformer (GPT-like) using rotary positional encoding.

It was trained on ~2.7M molecules (~90M tokens), from the high-quality MOSES and ChEMBL datasets.
More information on the datasets [here](https://tdcommons.ai/generation_tasks/molgen/).

MolDecod achieves an impressive performance for its size.
On 10,000 generated molecules with temperature 0.7:
- Validity: 0.95
- Uniqueness: 0.95
- Diversity: 0.87
- Novelty: 0.93

___

## App

You can launch a streamlit app to interact with the model, after installing the requirements, with the following command:
```
streamlit run streamlit.py
```

You can generate molecules, choosing the start tokens, maximum sequence length and model temperature, as well as visualize their structure and molecular properties.

<img width="742" alt="image" src="https://github.com/user-attachments/assets/8b6fec5b-fef4-4475-8358-8475dee558f1">

<img width="742" alt="image" src="https://github.com/user-attachments/assets/bf5f396f-7618-4d56-aac1-56f444a8ef9a">
