import streamlit as st
from utils.tokenizer import load_tokenizer
from utils.model import DecoderOnlyTransformer
from utils.generate import generate_molecule_streaming
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Generate and Visualize Tokenization", page_icon="🔍")

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
@st.cache_resource
def load_sp_model():
    sp_model_path = 'models/moldecod_tokenizer.model'
    return load_tokenizer(sp_model_path)

sp = load_sp_model()

# Define the model architecture and load MolDecod
@st.cache_resource
def load_model():
    vocab_size = sp.get_piece_size()
    d_model = 256
    nhead = 4
    num_encoder_layers = 4
    dropout = 0.25

    model = DecoderOnlyTransformer(vocab_size, d_model, nhead, num_layers=num_encoder_layers, dropout=dropout)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    checkpoint_path = 'models/moldecod_transformer.pth'
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer

model, optimizer = load_model()

st.title('Visualize Tokenization')
st.markdown('Generate SMILES strings and see how they are tokenized')

# Function to tokenize and display results
def tokenize_and_display(smiles):
    tokens = sp.encode(smiles, out_type=str)
    token_ids = [sp.piece_to_id(token) for token in tokens]
    
    # Create a color map
    n_colors = len(set(token_ids))
    color_palette = sns.color_palette("husl", n_colors=n_colors)
    color_map = {id: color_palette[i] for i, id in enumerate(set(token_ids))}
    
    # Create the visualization
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_ylim(0, 1)
    ax.set_xlim(0, len(smiles))
    ax.axis('off')
    
    for i, (token, token_id) in enumerate(zip(tokens, token_ids)):
        start = sum(len(t) for t in tokens[:i])
        end = start + len(token)
        ax.axvspan(start, end, facecolor=color_map[token_id], alpha=0.3)
        ax.text((start + end) / 2, 0.5, token, ha='center', va='center', fontsize=10, wrap=True)
    
    st.pyplot(fig)
    
    # Display statistics
    # st.write('Statistics:')
    col1, col2, col3 = st.columns(3)
    col1.metric("Number of tokens", len(tokens))
    col2.metric("Unique tokens", len(set(tokens)))
    col3.metric("Vocabulary size", sp.get_piece_size())
    
    # Display token details in an expander
    with st.expander("View Token Details"):
        df = pd.DataFrame({
            'Token': tokens,
            'ID': token_ids
        })
        st.table(df)

st.markdown('###')

# User inputs
start_tokens = st.text_input('Start tokens (e.g., C, CCN)', value='C')
col1, col2 = st.columns(2)
with col1:
   max_length = st.slider('Max length', min_value=10, max_value=1000, value=150)
with col2:
    temperature = st.slider('Temperature', min_value=0.01, max_value=1.0, value=0.5, step=0.01)

if st.button('Generate and Tokenize SMILES'):
    # Generate SMILES
    start_seq = sp.encode(start_tokens)
    start_seq = torch.tensor(start_seq, device=device)

    st.markdown('####')
    
    # Create a placeholder
    st.markdown('#### Generated molecule')
    output_placeholder = st.empty()
    
    with st.spinner('Generating...'):
        generated_molecule = generate_molecule_streaming(model, start_seq, sp, device, max_length=max_length, temperature=temperature, streamlit_context=True, output_placeholder=output_placeholder)
        
    st.markdown('#### Tokenization')
    # Tokenize and visualize
    tokenize_and_display(generated_molecule)

# Add a link back to the home page
st.sidebar.info("Return to 'Home' for molecule generation with structure visualization.")