# MolDecod-molecule-generation-tranformer
This repository contains MolDecod, a small transformer for SMILES molecule generation, its tokenizer, as well as the notebooks to train and use it.

MolDecod is a 5M-parameter decoder-only transformer (GPT-like) using rotary positional encoding.
It was train on ~2.7M molecules, from the MOSES and ChEMBL datasets.

MolDecod achieves an impressive performance for its size.
On 10,000 generated molecules:
- Validity: 0.95
- Uniqueness: 0.95
- Diversity: 0.87
- Novelty: 0.93