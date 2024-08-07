import torch
from utils.model import create_mask
from IPython.display import clear_output, display

def generate_molecule(model, start_seq, sp_model, device, max_length=150, temperature=0.7):
    model.eval()
    with torch.no_grad():
        current_seq = start_seq.to(device).unsqueeze(0)  # Add batch dimension
        for _ in range(max_length):
            src_mask = create_mask(current_seq.size(1)).to(device)
            output = model(current_seq, src_mask)
            logits = output[0, -1, :] / temperature  # Select last time step
            next_token_idx = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
            
            if next_token_idx == sp_model.piece_to_id('<EOS>'):
                break

            next_token_tensor = torch.tensor([[next_token_idx]], device=device)
            current_seq = torch.cat([current_seq, next_token_tensor], dim=1)
    
    # Decode using the tokenizer
    generated_sequence = sp_model.decode(current_seq[0].cpu().tolist())
    return generated_sequence.replace('<SOS>', '', 1)

def generate_molecule_streaming(model, start_seq, sp_model, device, max_length=150, temperature=0.7, streamlit_context=False, output_placeholder=None):
    model.eval()
    with torch.no_grad():
        current_seq = start_seq.to(device).unsqueeze(0)  # Add batch dimension
        generated_tokens = current_seq[0].cpu().tolist()  # Initialize with the start sequence tokens
        
        for _ in range(max_length):
            src_mask = create_mask(current_seq.size(1)).to(device)
            output = model(current_seq, src_mask)
            logits = output[0, -1, :] / temperature  # Select last time step
            next_token_idx = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
            
            if next_token_idx == sp_model.piece_to_id('<EOS>'):
                break

            # Add the token to the list
            generated_tokens.append(next_token_idx)

            # Display the current sequence
            current_seq_display = sp_model.decode(generated_tokens)
            if streamlit_context and output_placeholder is not None:
                output_placeholder.text(current_seq_display.replace('<SOS>', '', 1))
                output_placeholder.markdown(f"<h5>{current_seq_display.replace('<SOS>', '', 1)}</h5>", unsafe_allow_html=True)
            else:
                clear_output(wait=True)
                display(current_seq_display.replace('<SOS>', '', 1))

            next_token_tensor = torch.tensor([[next_token_idx]], device=device)
            current_seq = torch.cat([current_seq, next_token_tensor], dim=1)
    
    # Decode using the tokenizer
    generated_sequence = sp_model.decode(current_seq[0].cpu().tolist())
    return generated_sequence.replace('<SOS>', '', 1)
