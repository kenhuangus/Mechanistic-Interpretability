# Mechanistic Interpretability Demo: Sequence Reversal with MLP

This demo introduces data science students to Mechanistic Interpretability (MI) by training a simple Multi-Layer Perceptron (MLP) to reverse sequences (e.g., "abc" to "cba"). We then analyze the model's internals to understand how it learns and performs the task.

## What is Mechanistic Interpretability?

Mechanistic Interpretability aims to understand the internal mechanisms of machine learning models, particularly neural networks. Instead of treating models as black boxes, we reverse-engineer their computations to see how they process inputs and generate outputs.

Key concepts in MI:
- **Circuit Discovery**: Finding functional subnetworks within larger models.
- **Probing**: Using small models to interpret activations at intermediate layers.
- **Mechanisms**: Understanding how models compute answers, not just what they predict.

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

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the demo:
   ```bash
   python demo.py
   ```

The script will:
- Generate training data.
- Train the model (may take 1-2 minutes).
- Display training loss plot.
- Analyze a sample prediction and show activation maps.
- Print explanations.

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

## Extending the Demo

- **Transformer Version**: Replace MLP with a small transformer and visualize attention matrices.
- **More Complex Tasks**: Try arithmetic, logic, or natural language understanding.
- **Probing Layer**: Train probes on hidden activations to predict intermediate computations.
- **Interventions**: Modify activations and see how predictions change.

This demo provides a gentle introduction to MI concepts. In practice, MI requires careful experimentation and often custom models or tools to extract meaningful insights from large language models.
