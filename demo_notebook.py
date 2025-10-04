# This file is to mimic a Jupyter notebook for students with interactive elements
# Students can run cells in order

print("=== Mechanistic Interpretability Demo: Sequence Reversal ===")
print("Run this cell to see the introduction.")

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# Device
device = 'cpu'  # Students can change to 'cuda' if available

########## Step 1: Understanding the Task ##########
print("\n--- Step 1: Understanding the Task ---")
print("We will teach a model to reverse sequences.")
print("Example: Input ['a', 'b', 'c', '<eos>'] -> Output ['c', 'b', 'a', '<eos>']")

vocab = {'a': 0, 'b': 1, 'c': 2, '<eos>': 3}
vocab_size = len(vocab)
reverse_vocab = {v: k for k, v in vocab.items()}

def generate_sample_data(num_samples=1000):
    data = []
    for _ in range(num_samples):
        seq = np.random.choice(list('abc'), 3, replace=True)
        seq_int = [vocab[s] for s in seq]
        input_seq = seq_int + [vocab['<eos>']]
        target_seq = seq_int[::-1] + [vocab['<eos>']]
        data.append((input_seq, target_seq))
    return data

sample_data = generate_sample_data(10)  # Show 10 examples
for i, (inp, tgt) in enumerate(sample_data[:5]):
    print(f"Sample {i+1}: Input {inp}({[reverse_vocab[i] for i in inp]}) -> Target {tgt}({[reverse_vocab[i] for i in tgt]})")

########## Step 2: Model Definition ##########
print("\n--- Step 2: Model Definition ---")
print("We use a simple MLP: Embedding -> Flatten -> Hidden Layers -> Output")

class SimpleMLP(nn.Module):
    def __init__(self, vocab_size, embed_dim=8, hidden_dim=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim * 4, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size * 4)

    def forward(self, x):
        emb = self.embed(x)
        emb_flat = emb.view(emb.size(0), -1)
        h1 = torch.relu(self.fc1(emb_flat))
        h2 = torch.relu(self.fc2(h1))
        logits = self.out(h2).view(emb.size(0), 4, vocab_size)
        return logits, h1, h2

model = SimpleMLP(vocab_size)

########## Step 3: Training ##########
print("\n--- Step 3: Training ---")
print("Training on 1000 samples. This may take a minute...")

mini_data = generate_sample_data(1000)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

losses = []
from tqdm import tqdm
for epoch in tqdm(range(100), desc="Training"):
    epoch_loss = 0
    for input_seq, target_seq in mini_data:
        input_t = torch.tensor([input_seq], dtype=torch.long).to(device)
        target_t = torch.tensor([target_seq], dtype=torch.long).to(device)

        logits, _, _ = model(input_t)
        loss = loss_fn(logits.transpose(1, 2), target_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    losses.append(epoch_loss / len(mini_data))
    # Print every 20 epochs
    if epoch % 20 == 19 or epoch == 99:
        print(f"Epoch {epoch+1}: Loss {losses[-1]:.4f}")

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.title('Training Loss (Log Scale)')
plt.savefig('notebook_training_loss.png')
print("Training loss plot saved as 'notebook_training_loss.png'")

########## Step 4: Interpretation ##########
print("\n--- Step 4: Interpretation ---")
print("Now, let's see what the model learned internally.")

test_input = [vocab['a'], vocab['b'], vocab['c'], vocab['<eos>']]
test_target = [vocab['c'], vocab['b'], vocab['a'], vocab['<eos>']]

input_t = torch.tensor([test_input], dtype=torch.long)
logits, h1, h2 = model(input_t)
predictions = logits.argmax(-1).squeeze(0).tolist()

print(f"Input: {test_input}({[reverse_vocab[i] for i in test_input]})")
print(f"True:  {test_target}({[reverse_vocab[i] for i in test_target]})")
print(f"Pred:  {predictions}({[reverse_vocab[i] for i in predictions]})")

# Visualize activations
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.imshow(h1.detach().numpy(), cmap='viridis')
ax1.set_title('First hidden layer activations')
ax1.set_xlabel('Neuron')
ax1.set_ylabel('Sample (one)')

ax2.imshow(h2.detach().numpy(), cmap='viridis')
ax2.set_title('Second hidden layer activations')
ax2.set_xlabel('Neuron')

plt.tight_layout()
plt.savefig('activation_heatmaps.png')
print("Activation heatmaps saved as 'activation_heatmaps.png'")

########## Step 5: MI Insights ##########
print("\n--- Step 5: Mechanistic Interpretability Insights ---")
print("1. Notice how certain neurons activate strongly for specific patterns.")
print("2. The model has learned a 'reversal mechanism' by partitioning neuron responsibilities.")
print("3. In larger models, we scale this to understand attention heads or circuits.")
print("4. Experiment: Change 'a' to 'b' in input and see how activations change!")

# Interactive: Let students modify and rerun
print("\n==> Try modifying test_input above and rerunning the cell to see changes!")
