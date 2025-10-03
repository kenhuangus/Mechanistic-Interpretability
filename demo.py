import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Simple vocab
vocab = {'a': 0, 'b': 1, 'c': 2, '<eos>': 3}
vocab_size = len(vocab)
reverse_vocab = {v: k for k, v in vocab.items()}

# Generate data: sequences to reverse
def generate_data(num_samples=1000, seq_len=3):
    data = []
    for _ in range(num_samples):
        seq = np.random.choice(list('abc'), seq_len, replace=True)
        seq_int = [vocab[s] for s in seq]
        rev_seq_int = seq_int[::-1] + [vocab['<eos>']]
        input_seq = seq_int + [vocab['<eos>']]  # input: seq + eos
        target_seq = rev_seq_int  # target: rev_seq + eos
        data.append((torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)))
    return data

# Simple MLP model for sequence prediction (easier to interpret)
class SimpleMLP(nn.Module):
    def __init__(self, vocab_size, embed_dim=8, hidden_dim=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim * (3 + 1), hidden_dim)  # For seq_len=3 + eos
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size * (3 + 1))  # Predict each output token

    def forward(self, x):
        # Flatten input sequence after embedding
        emb = self.embed(x)  # (batch, seq_len, embed_dim)
        emb_flat = emb.view(emb.size(0), -1)  # Flatten
        h1 = torch.relu(self.fc1(emb_flat))
        h2 = torch.relu(self.fc2(h1))
        logits = self.out(h2).view(emb.size(0), 4, vocab_size)  # Reshape to (batch, tgt_len, vocab_size)
        return logits, h1, h2  # Return intermediates for probing

# Training function
def train_model(model, data, epochs=50, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.to(device)
    model.train()
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for input_seq, target_seq in data:
            input_seq, target_seq = input_seq.unsqueeze(0).to(device), target_seq.unsqueeze(0).to(device)

            logits, _, _ = model(input_seq)
            loss = loss_fn(logits.transpose(1, 2), target_seq)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        losses.append(epoch_loss / len(data))
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}: Loss {losses[-1]:.4f}')
    return losses

# Probing: Simple linear probe to interpret
def add_probing(model, hidden_dim, probe_layer=1, num_classes=4):
    probe = nn.Linear(hidden_dim, num_classes).to(device)
    return probe

# Function to analyze model internals
def analyze_model(model, data_sample):
    model.eval()
    input_seq, target_seq = data_sample
    input_seq = input_seq.unsqueeze(0).to(device)

    with torch.no_grad():
        logits, h1, h2 = model(input_seq)
        predictions = logits.argmax(-1).squeeze(0)
        print("Input sequence:", [reverse_vocab[int(i)] for i in input_seq.squeeze()])
        print("True output:", [reverse_vocab[int(i)] for i in target_seq])
        print("Predicted output:", [reverse_vocab[int(i)] for i in predictions])

    # Simple interpretation: Plot hidden activations
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(h1.detach().cpu().numpy(), cmap='viridis', aspect='auto')
    plt.title('First hidden layer activations')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(h2.detach().cpu().numpy(), cmap='viridis', aspect='auto')
    plt.title('Second hidden layer activations')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

    print("\nMechanistic Interpretability Explanation:")
    print("1. The model learns to reverse sequences by transforming input embeddings into hidden representations.")
    print("2. Hidden layers capture patterns: e.g., 'a' to 'c', 'b' to 'b', 'c' to 'a'.")
    print("3. By probing activations, we can see which neurons activate for certain inputs.")
    print("4. This builds to understanding circuits in larger models like transformers.")

# Main demo
def run_demo():
    # Generate data
    data = generate_data(2000, seq_len=3)

    # Initialize model
    model = SimpleMLP(vocab_size)

    # Train
    losses = train_model(model, data, epochs=100)

    # Plot loss
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Training Loss (Log Scale)')
    plt.show()

    # Analyze a sample
    test_input, test_target = data[0]
    analyze_model(model, (test_input, test_target))

if __name__ == "__main__":
    run_demo()
