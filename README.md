# Mechanistic Interpretability Demo: Sequence Reversal with MLP

This demo introduces data science students to Mechanistic Interpretability (MI) by training a simple Multi-Layer Perceptron (MLP) to reverse sequences (e.g., "abc" to "cba"). We then analyze the model's internals to understand how it learns and performs the task.

## What is Mechanistic Interpretability?

Mechanistic Interpretability (MI) is an emerging field that aims to reverse-engineer the internal computations of neural networks to understand how they work, rather than just what they do. Unlike traditional interpretability methods that focus on feature importance or saliency maps, MI seeks to discover the exact algorithms and circuits inside models.

### Why MI Matters
- **Model Reliability**: Understands why models fail and ensures they follow intended logic
- **Safety & Alignment**: Critical for AI safety research, especially in large language models
- **Scientific Progress**: Advances our fundamental understanding of neural computation
- **Robustness**: Identifies and fixes model vulnerabilities or biases

### Key MI Concepts
- **Circuit Discovery**: Identifying functional subnetworks (circuits) within large models that handle specific computations
- **Probing**: Training small classifiers to interpret hidden layer activations
- **Mechanisms**: Understanding the concrete algorithms implemented by model weights
- **Activation Engineering**: Techniques to manipulate internal computations for understanding or intervention
- **Superposition & Polysemanticity**: How models pack multiple concepts into single neuron activations

### MI Research Areas
- **Toy Models**: Small, interpretable models (like this MLP) that learn simple tasks
- **Grokking**: Sudden generalization after overfitting, revealing internal learning dynamics
- **Sparse Autoencoders**: Unpacking dense activations into interpretable feature sets
- **Path Patching**: Causal analysis of information flow through model layers
- **Representation Engineering**: Steerable representations for model behavior control

### Connection to Transformers and LLMs
This demo uses a simple MLP, but MI techniques scale to transformers:
- Attention heads implement specific "algorithms" (e.g., copying, induction, math)
- Circuits span multiple heads and layers
- MLP sublayers perform feature processing and transformations
- Understanding these mechanisms helps with jailbreak resistance, hallucination reduction, and capability elicitation

## Demo Overview

### The Task
The model learns to reverse short sequences of letters ('a', 'b', 'c') followed by an end token ('<eos>').

Example:
- Input: ['a', 'b', 'c', '<eos>']
- Target Output: ['c', 'b', 'a', '<eos>']

### Model Architecture
We use a simple MLP:
1. **Embedding Layer**: Converts token IDs to vectors.
2. **Flattened Input**: Inputs are concatenated into a single vector.
3. **Hidden Layers**: Two fully connected layers with ReLU activation.
4. **Output Layer**: Predicts the entire output sequence at once.

### Training
- Dataset: 2000 randomly generated sequences.
- Optimizer: Adam with learning rate 0.001.
- Loss: Cross-Entropy loss for sequence prediction.

### Interpretation
After training, we:
1. **Activate Sampling**: Examine model predictions on a test input.
2. **Activation Probing**: Visualize neuron activations in hidden layers.
3. **Mechanistic Understanding**: Draw connections to larger models like transformers.

## How to Run the Demo

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/kenhuangus/Mechanistic-Interpretability.git
   cd Mechanistic-Interpretability
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Scripts

#### For Educational Walkthrough: `demo_notebook.py`
- **Best For**: Students learning step-by-step, beginners, interactive exploration
- **Features**: Detailed explanations, progress bars, educational guidance, modifiable for experiments
- **Run**:
  ```bash
  python demo_notebook.py
  ```
- **Output**: Step-by-step console tutorial, same plots as main demo, encouragement to modify code

#### For Automated Demo: `demo.py`
- **Best For**: Overview of results, automated run, advanced users
- **Features**: End-to-end execution, brief logs, focused on results
- **Run**:
  ```bash
  python demo.py
  ```
- **Output**: Training epochs, final analysis, generated PNG plots

### Expected Behavior
Both scripts will:
- Generate 2000 training sequences for `demo.py` (1000 for notebook)
- Train for 100 epochs to perfect accuracy (loss → 0.0000)
- Save `training_loss.png` and `activation_heatmaps.png`
- Print MI explanations and provide perfect predictions

## Expected Output
- Training loss decreases rapidly to near zero.
- Perfect predictions on the test sample.
- Heatmaps showing which neurons activate for different inputs.
- Explanatory text connecting to MI principles.

## Educational Takeaways

1. **Model Internals Matter**: Even simple models have interpretable internal computations.
2. **Probing Techniques**: Small classifiers or visualizations can reveal how information flows.
3. **Scaling to Complex Models**: This foundation helps understand attention mechanisms, superposition, and circuits in transformers.
4. **Research Directions**: MI research focuses on automating circuit discovery, understanding emerging abilities, and ensuring model safety.

## Further Reading

- Anthropic's Mechanical Interpretability Blog: https://transformer-circuits.pub/
- Neel Nanda's Explaining Transformers: https://www.3blue1brown.com/lessons/attention
- "Circuits" Paper: https://arxiv.org/abs/2303.09740
- "A Mathematical Framework for Transformer Circuits": https://arxiv.org/abs/2305.00010
- AI Safety Camp: https://aisafety.camp/

## Extending the Demo

- **Transformer Version**: Replace MLP with a small transformer and visualize attention matrices.
- **More Complex Tasks**: Try arithmetic, logic, or natural language understanding.
- **Probing Layer**: Train probes on hidden activations to predict intermediate computations.
- **Interventions**: Modify activations and see how predictions change.

This demo provides a gentle introduction to MI concepts. In practice, MI requires careful experimentation and often custom models or tools to extract meaningful insights from large language models.
