import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from utils.tokenizer import load_tokenizer
from utils.model import DecoderOnlyTransformer
from utils.generate import generate_molecule_streaming
from utils.properties import calculate_properties, draw_molecule

st.set_page_config(page_title="MolDecod App", page_icon="🧪")

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

# Streamlit app
st.title('MolDecod - Molecule Generation')
st.markdown('Generate molecules with a decoder-only transformer model')

st.markdown('###')

# User inputs
start_tokens = st.text_input('Start tokens (e.g., C, CCN)', value='C')
col1, col2 = st.columns(2)
with col1:
    max_length = st.slider('Max length', min_value=10, max_value=1000, value=150)
with col2:
    temperature = st.slider('Temperature', min_value=0.01, max_value=1.0, value=0.5, step=0.1)

if st.button('Generate Molecule'):
    # Correctly handle start tokens
    start_seq = sp.encode(start_tokens)
    start_seq = torch.tensor(start_seq, device=device)

    st.markdown('###')
    
    # Create a placeholder
    st.markdown('#### Generated molecule')
    output_placeholder = st.empty()
    
    with st.spinner('Generating...'):
        generated_molecule = generate_molecule_streaming(model, start_seq, sp, device, max_length=max_length, temperature=temperature, streamlit_context=True, output_placeholder=output_placeholder)
    
    st.markdown('')

    # Draw molecule
    img = draw_molecule(generated_molecule)
    if img:
        st.image(img, caption='Generated Molecule Structure')
    
    # Calculate properties
    properties = calculate_properties(generated_molecule)
    if properties:
        st.write('Molecular Properties:')
        property_data = [[prop, value] for prop, value in properties.items()]
        property_df = pd.DataFrame(property_data, columns=['Property', 'Value'])
        st.table(property_df)
    else:
        st.write("Invalid molecule")

# Add information about the other page
st.sidebar.info("Go to 'Visualize Tokenization' to see how SMILES strings are tokenized.")