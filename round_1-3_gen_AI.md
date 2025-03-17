# Short Answers to Machine Learning Questions

## Transformer Architecture
The transformer architecture consists of an encoder-decoder structure built on self-attention mechanisms. Key components include multi-head attention, feed-forward networks, residual connections, layer normalization, and positional encodings. It processes all input tokens in parallel rather than sequentially, enabling more efficient training.

## Fine-tuning Open Source Models
1. Select a pre-trained model relevant to your task
2. Prepare your custom dataset in the required format
3. Configure hyperparameters (learning rate, batch size, etc.)
4. Initialize with pre-trained weights
5. Train on your dataset for fewer epochs
6. Evaluate and adjust as needed

## Self-Attention
Self-attention allows a model to weigh the importance of different tokens in the input sequence when encoding each token. It works by computing query, key, and value vectors for each token, then using similarity between queries and keys to determine attention weights. It's effective because it captures long-range dependencies and contextual relationships without distance limitations.

## Embeddings
Embeddings are dense vector representations of discrete tokens (words, characters). We create them through:
1. Random initialization then learned during training
2. Pre-training methods like Word2Vec, GloVe, or BERT-style masked language modeling
3. Contextual embedding models that capture semantic relationships

## Reducing Model Size (70B to 8B)
1. Knowledge distillation: Train smaller model to mimic larger one
2. Pruning: Remove less important weights
3. Quantization: Reduce precision of weights
4. Low-rank factorization of weight matrices
5. Parameter sharing across layers
6. Architectural modifications (reducing layers, attention heads)

## Evaluation Metrics for Fine-tuned Models
1. Perplexity: Measures prediction quality
2. BLEU, ROUGE, METEOR: For text generation tasks
3. Accuracy, Precision, Recall, F1: For classification
4. GLUE/SuperGLUE benchmarks: For general language understanding
5. Task-specific metrics (e.g., exact match for QA)

## Quantization Techniques
1. Post-training quantization: Convert weights after training
2. Quantization-aware training: Train with simulated quantization
3. INT8/INT4 quantization: Reduce 32-bit float to 8/4-bit integers
4. Weight-only vs. activation quantization
5. Mixed-precision approaches (different precision for different layers)

## Multihead Attention and Cross-Attention
Multihead attention splits attention into multiple "heads" that learn different patterns, then combines results. Cross-attention computes attention between two different sequences (e.g., source and target in translation), allowing one sequence to attend to another.

## Fine-tuning Challenges and Strategies
Challenges:
- Catastrophic forgetting
- Overfitting on small datasets
- Computational resources

Strategies:
- Gradual unfreezing of layers
- Parameter-efficient fine-tuning (LoRA, adapters)
- Learning rate scheduling
- Early stopping
- Data augmentation
- Mixtures of experts

## Regularization Techniques
- Dropout: Randomly deactivates neurons during training
- Label smoothing: Softens one-hot encodings to prevent overconfidence
- Weight decay: Penalizes large weights to reduce model complexity
- Gradient clipping: Prevents exploding gradients

## NLP Loss Functions
- Cross-entropy loss: Standard for next token prediction
- KL divergence: For knowledge distillation
- Contrastive loss: For semantic similarity tasks
- Triplet loss: For learning embeddings
- REINFORCE/PPO: For reinforcement learning from human feedback
