# MolDecod: Decoder-only transformer for molecule generation

This repository contains:
- MolDecod, a small transformer for SMILES molecule generation
- its tokenizer, a SentencePiece model with a vocab size of 1000
- the notebooks to train and use it

MolDecod is a 5M-parameter decoder-only transformer (GPT-like) using rotary positional encoding.

It was trained on ~2.7M molecules (~90M tokens), from the high-quality MOSES and ChEMBL datasets. More information on the datasets [here](https://tdcommons.ai/generation_tasks/molgen/).

MolDecod achieves an impressive performance for its size.
On 10,000 generated molecules:
- Validity: 0.95
- Uniqueness: 0.95
- Diversity: 0.87
- Novelty: 0.93

___

You can install the requirements and launch the streamlit app to interact with the model:
```
streamlit run streamlit.py
```
in the shell at the root folder.

<img width="428" alt="image" src="https://github.com/user-attachments/assets/1d5d50d4-94b1-47ae-b9d1-147dffe92b3a">

<img width="384" alt="image" src="https://github.com/user-attachments/assets/facafc0d-d4e3-4c65-83d9-04b5d51fd9c7">
