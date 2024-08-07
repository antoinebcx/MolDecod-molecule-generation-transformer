# MolDecod: Decoder-only transformer for molecule generation

This repository contains:
- MolDecod, a small transformer for SMILES molecule generation
- its tokenizer, a SentencePiece model with a vocab size of 1000
- the notebooks to train and use it

MolDecod is a 5M-parameter decoder-only transformer (GPT-like) using rotary positional encoding.

It was trained on ~2.7M molecules, from the high-quality MOSES and ChEMBL datasets (more information [here](https://tdcommons.ai/generation_tasks/molgen/)).

MolDecod achieves an impressive performance for its size.
On 10,000 generated molecules:
- Validity: 0.95
- Uniqueness: 0.95
- Diversity: 0.87
- Novelty: 0.93