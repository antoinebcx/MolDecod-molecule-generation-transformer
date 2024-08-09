# MolDecod technical report

## Introduction

MolDecod is a decoder-only transformer model (GPT-like) using rotary positional encoding designed for molecular representation learning. It efficiently encodes and generates SMILES strings, a standard format for representing molecular structures.

This report details MolDecod's architecture, training, and evaluation, demonstrating its effectiveness in producing chemically valid, diverse, and novel molecules. Through a custom SentencePiece tokenizer and targeted training, MolDecod proves to be a powerful model for its size.

## Data

We train the model on a combination of two datasets for molecule generation:
- MOSES, a benchmark platform for distribution learning based molecule generation, processed from the ZINC Clean Leads dataset, containing ~ 2 million molecules.
- ChEMBL, a manually curated database of bioactive molecules with drug-like properties, containing ~ 2 million molecules.

The random split results in 2.7 million molecules in the training set (70%), and ~600k molecules in both the validation and test sets (15% each).

The data was downloaded using the [TDCommons library](https://tdcommons.ai/generation_tasks/molgen/).

## Tokenizer

📖 [SentencePiece](https://github.com/google/sentencepiece), [tokenizer.py](https://github.com/tonito9/MolDecod-molecule-generation-transformer/blob/main/utils/tokenizer.py) & [training notebook](https://github.com/tonito9/MolDecod-molecule-generation-transformer/blob/main/notebooks/eval_moldecod.ipynb)

This project uses a custom SentencePiece tokenizer to process SMILES strings for molecular representation learning. The tokenizer is trained with a 1,000-token vocabulary, including special tokens (< SOS >, < EOS >, < PAD >). It converts SMILES into subword units, ensuring robust encoding of molecular structures.


## Architecture

📖 [model.py](https://github.com/tonito9/MolDecod-molecule-generation-transformer/blob/main/utils/model.py)

### Key features

MolDecod is a decoder-only transformer model, with the following specifications:
- Embedding Layer: Converts input tokens to 256-dimensional vectors.
- Rotary Positional Encoding: Implements position-aware representations without additional parameters.
- Transformer Decoder Blocks: Four identical layers, each containing:
    - Multi-head Self-Attention mechanism (4 heads)
    - Position-wise Feed-Forward Network (FFN)
    - Layer Normalization (applied first in each sub-layer)
- Output Layer: Linear projection to vocabulary size for token prediction.

This results, with the buffers, in a total of ~5 million parameters, including ~4 million trainable ones.

The model uses GELU activation functions and employs a dropout rate of 0.25 for regularization. It generates a causal attention mask to ensure autoregressive behavior during training and inference.

### Diagram

                     +--------------------------+
                     |      Input Tokens        |
                     +--------------------------+
                                 |
                                 v
          +--------------------------------------------+
          |            Embedding Layer                |
          |  - Converts tokens to 256-D dense vectors |
          |  - Scales by sqrt(d_model)                |
          +--------------------------------------------+
                                 |
                                 v
          +--------------------------------------------+
          |    Rotary Positional Encoding Layer        |
          |  - Applies rotary sinusoidal encoding      |
          |  - Separates and combines even/odd indices |
          +--------------------------------------------+
                                 |
                                 v
        +------------------------------------------------------+
        |  Transformer Block (Repeated 4 Times)                |
        |                                                      |
        |  +-----------------+   +-------------------------+   |
        |  | Multi-Head       |   | Feed-Forward            |  |
        |  | Self-Attention   |   | Network with GELU       |  |
        |  |  - 4 heads       |   |  - 1024-D hidden layer  |  |
        |  |  - Scaled dot-   |   |  - Dropout (0.25)       |  |
        |  |    product       |   |  - Layer Normalization  |  |
        |  +-----------------+   +-------------------------+   |
        |        |                           |                 |
        |        v                           v                 |
        |    Residual                       Residual           |
        |    Connection                    Connection          |
        +------------------------------------------------------+
                                 |
                                 v
          +--------------------------------------------+
          |            Linear Layer                    |
          |  - Maps 256-D vectors to vocab size (1000) |
          +--------------------------------------------+
                                 |
                                 v
          +--------------------------------------------+
          |             Dropout Layer                  |
          |  - Applies dropout (0.25) to output        |
          +--------------------------------------------+
                                 |
                                 v
          +--------------------------------------------+
          |           Masking (Causal Mask)            |
          |  - Prevents future tokens from attending   |
          |    to current/past tokens                  |
          +--------------------------------------------+
                                 |
                                 v
          +--------------------------------------------+
          |             Output Logits                  |
          |  - Softmax for probability distribution    |
          +--------------------------------------------+
                                 |
                                 v
          +--------------------------------------------+
          |           Token Selection & Update         |
          |  - Sample next token based on probabilities|
          |  - Append to sequence for next iteration   |
          +--------------------------------------------+
                                 |
                                 v
          +--------------------------------------------+
          |            Decoded Output Sequence         |
          |  - Decode generated tokens back to text    |
          +--------------------------------------------+


## Training

📖 [training notebook](https://github.com/tonito9/MolDecod-molecule-generation-transformer/blob/main/notebooks/eval_moldecod.ipynb)

The model is optimized using the AdamW optimizer with a learning rate of 1e-4 and gradient clipping set at 1.0 to prevent gradient explosion. Training occurs over 8 epochs with a batch size of 64. CrossEntropyLoss is used to assess the difference between predicted and actual token sequences, while the OneCycleLR scheduler dynamically adjusts the learning rate, peaking early and then gradually decreasing throughout training.

During the training phase, a causal mask is applied to maintain autoregressive sequence generation, and gradients are clipped to ensure stability. After each epoch, the model is evaluated on a validation set to monitor performance. The model with the lowest validation loss is saved as the best model. This process ensures the model is well-regularized, efficiently trained, and robust for molecular representation tasks.


## Evaluation

📖 [evaluation notebook](https://github.com/tonito9/MolDecod-molecule-generation-transformer/blob/main/notebooks/eval_moldecod.ipynb)

### Metrics

We 10,000 generate molecules for different temperature levels and evaluate the model on the following metrics:
- **Validity**: This measures the proportion of generated molecules that are chemically valid.
- **Uniqueness**: The proportion of unique molecules among the valid ones.
- **Diversity**: A measure of structural diversity among the generated molecules.
- **Novelty**: The proportion of generated molecules not present in the training set.
- **KL Divergence**: Measures how much the distribution of generated molecules differs from the training set.
- **Fragment Similarity**: Similarity of generated molecules to the training set based on molecular fragments.
- **Scaffold Diversity**: Measures the diversity of molecular scaffolds in the generated set.

### Results

| Temperature | Validity | Uniqueness | Diversity | Novelty | KL Divergence | Fragment Similarity | Scaffold Diversity |
|-------------|----------|------------|-----------|---------|---------------|---------------------|--------------------|
| 0.1         | 1.00     | 0.04       | 0.76      | 0.9455  | 6.4742        | 0.0545              | 0.0148             |
| 0.25        | 1.00     | 0.49       | 0.81      | 0.8347  | 4.3664        | 0.1653              | 0.1398             |
| 0.5         | 0.98     | 0.95       | 0.85      | 0.8768  | 5.7033        | 0.1237              | 0.4556             |
| 0.7         | 0.96     | 0.95       | 0.87      | 0.9240  | 5.6936        | 0.0778              | 0.6540             |
| 0.9         | 0.88     | 0.88       | 0.88      | 0.9562  | 5.3179        | 0.0502              | 0.7524             |


MolDecod demonstrates robust performance across various temperature settings for its size. It achieves high validity (88-100%) and novelty (83-96%) across all temperatures. Lower temperatures (0.1-0.25) excel in generating valid molecules highly similar to the training set, as evidenced by low uniqueness (4-49%) and scaffold diversity (1-14%). Mid-range temperatures (0.5-0.7) offer a balance between validity (96-98%) and diversity, with high uniqueness (95%) and improved scaffold diversity (46-65%). The highest temperature (0.9) maximizes novelty (96%), diversity (88%), and scaffold diversity (75%), albeit with a slight decrease in validity (88%).

The model exhibits a clear trade-off between molecular validity and structural diversity as temperature increases. This allows for flexible application: lower temperatures for generating analogs of known compounds, mid-range temperatures for a balance of validity and novelty, and higher temperatures for maximal exploration of chemical space. The KL divergence and fragment similarity metrics further support these observations, with the closest match to the training distribution at 0.25 temperature.


## Conclusion

MolDecod successfully balances the generation of valid, novel, and diverse molecules using its transformer-based architecture. The model adapts well across temperature settings, offering flexibility from generating known analogs to exploring novel chemical spaces.
