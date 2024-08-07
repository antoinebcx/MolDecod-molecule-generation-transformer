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
On 10,000 generated molecules:
- Validity: 0.95
- Uniqueness: 0.95
- Diversity: 0.87
- Novelty: 0.93

___

## UI

You can launch a streamlit app to interact with the model, after installing the requirements, with the following command:
```
streamlit run streamlit.py
```


<img width="429" alt="image" src="https://github.com/user-attachments/assets/1cb03cde-b2e8-4fb4-a33f-98954584919e">


<img width="384" alt="image" src="https://github.com/user-attachments/assets/facafc0d-d4e3-4c65-83d9-04b5d51fd9c7">
