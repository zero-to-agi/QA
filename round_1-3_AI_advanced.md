# AI & Machine Learning: Questions & Answers

## Domain Expertise - Artificial Intelligence

1. **What's the difference between narrow AI and general AI?**
   - Narrow AI excels at specific tasks (like chess or image recognition) but can't transfer learning. General AI would have human-like ability to learn and apply knowledge across domains. Currently, only narrow AI exists in practice.

2. **How do neural networks "learn"?**
   - They adjust connection weights between neurons through backpropagation, minimizing the difference between predicted and actual outputs using gradient descent algorithms.

3. **What is the "black box" problem in AI?**
   - Many AI systems, especially deep learning models, make decisions that humans can't easily interpret or explain, creating challenges for trust, accountability, and debugging.

4. **What are the key differences between symbolic AI and machine learning?**
   - Symbolic AI uses explicit rules and logic coded by humans, while machine learning learns patterns from data without explicit programming.

5. **How has the AI alignment problem evolved?**
   - AI alignment has shifted from theoretical concerns to practical implementation challenges. Techniques like RLHF and constitutional AI show promise.

6. **What is explainable AI (XAI)?**
   - XAI focuses on creating models and techniques that enable humans to understand, trust, and effectively manage AI systems by making their decisions transparent and interpretable.

7. **How does federated learning differ from traditional centralized learning?**
   - Federated learning trains models across multiple devices without exchanging the actual data, only model updates, preserving privacy and reducing data transfer needs.

8. **What are the major approaches to knowledge representation in AI?**
   - Semantic networks, frames, rules, logic-based systems, and more recently, embeddings and knowledge graphs are all ways to represent knowledge in AI systems.

9. **What is the role of heuristics in AI problem-solving?**
   - Heuristics are "rules of thumb" that guide search or decision processes to find adequate solutions more efficiently than exhaustive approaches, trading some optimality for practicality.

10. **How does multi-agent AI differ from single-agent systems?**
    - Multi-agent systems involve multiple autonomous entities that interact, cooperate or compete, requiring coordination protocols, conflict resolution, and often game theory principles.

## Generative AI

1. **How do diffusion models generate images?**
   - They gradually add noise to images during training, then learn to reverse this process, generating new images by progressively denoising random noise.

2. **What is the key insight behind transformer models used in generative AI?**
   - Self-attention mechanisms that allow the model to weigh the importance of different input elements regardless of their distance from each other, capturing long-range dependencies.

3. **What are common sampling methods used in text generation?**
   - Greedy decoding (most probable token), beam search (maintaining multiple possibilities), temperature sampling (controlling randomness), and nucleus/top-p sampling (sampling from a dynamic subset).
온도와 top-p 함께 사용: 많은 시스템에서 두 기법을 함께 사용합니다. 먼저 온도로 전체 분포를 조정한 다음 top-p로 샘플링합니다.
ChatGPT와 같은 대화형 AI: 일반적으로 온도 값을 0.7-0.9, top-p 값을 0.9-0.95로 설정하여 자연스러우면서도 다양한 응답을 생성합니다.
코드 생성: 온도를 낮게(0.2-0.4) 설정하고 top-p를 0.9 이상으로 설정하여 결정적이지만 일부 유연성을 가진 코드를 생성합니다.
4. **How do diffusion models compare with GANs?**
   - GANs use two competing networks but can be unstable. Diffusion models gradually add and remove noise, typically producing higher quality and more diverse outputs.

5. **What architectural improvements enabled the transition from GPT-2 to GPT-3?**
   - Massive scaling of parameters (175B vs 1.5B), better training methods, and improved data filtering led to emergent capabilities in GPT-3.

6. **What is controllable generation in generative AI?**
   - Techniques that allow precise control over specific attributes of generated content, like style, sentiment, or specific features, through conditioning signals or specialized training.

7. **How do vector quantized models like DALL-E work?**
   - They compress images into discrete tokens (like words), allowing them to be processed by language models, then reconstruct images from these tokens.

8. **What is the role of latent spaces in generative models?**
   - **Latent spaces are compressed representations** where similar concepts are nearby, enabling operations like interpolation between concepts and controlled generation by navigating this space.

10. **What are common evaluation challenges for generative models?**
    - Lack of single objective metric, subjective quality assessment, balancing novelty vs. quality, and evaluating alignment with human intent or instruction.

## Large Language Models (LLMs)

1. **What is the scaling law for language models?**
   - Performance improves predictably as a power-law with increases in model size, dataset size, and compute—suggesting larger models continue to improve capabilities.

2. **How do LLMs handle multiple languages?**
   - Either through multilingual training data, specialized tokenization for different languages, or language-specific adaptations like embedding spaces.

3. **What is prompt engineering?**
   - The practice of designing effective input prompts that elicit desired behaviors from LLMs, **often using techniques like few-shot examples, specific instructions, or structured formats.**

4. **What are the key challenges in scaling language models?**
   - **Hardware limitations, increasing computational costs**, diminishing returns on some tasks, and **data exhaustion for high-quality training.**

5. **How do RLHF and constitutional AI improve alignment?**
   - These improve alignment by incorporating human feedback and pre-defined values, **helping models better understand human preferences and ethical boundaries.**

6. **What is retrieval-augmented generation (RAG)?**
   - A technique where LLMs are connected to **external knowledge sources that retrieve relevant information before generating responses**, reducing hallucinations and improving factuality.

7. **How do instruction tuning and chain-of-thought prompting improve LLM reasoning?**
   - Instruction tuning makes models better at following specific directions, while chain-of-thought prompts models to show step-by-step reasoning, improving complex problem-solving.

8. **What are the different ways to measure LLM capabilities?**
   - Benchmark suites (like MMLU, HellaSwag), human evaluations, application-specific metrics, adversarial testing, and real-world task performance.

9. **How do token merging techniques like BPE work in LLMs?**
   - They create vocabularies by iteratively merging common character pairs into single tokens, balancing vocabulary size with the ability to represent rare words.

10. **What are the differences between decoder-only, encoder-only, and encoder-decoder architectures?**
    - **Decoder-only (like GPT) generates text sequentially, encoder-only (like BERT) understands text bidirectionally, and encoder-decoder (like T5) transforms input to output sequence**s.

## Training & Serving Large ML Models

1. **What is model distillation and when is it useful?**
   - A technique where a smaller model learns to mimic a larger model, reducing deployment size while retaining much of the performance, useful for resource-constrained environments.

2. **How does quantization affect model performance?**
   - Reduces precision of model weights (e.g., from 32-bit to 8-bit), decreasing memory and computation needs, sometimes with minimal accuracy loss if done carefully.

3. **What are the benefits of precision scaling in training?**
   - Mixed precision (using lower precision like FP16 for most operations) reduces memory usage and speeds up training, especially on specialized hardware like GPUs.

4. **What strategies optimize training of large transformer models?**
   - Use mixed precision, gradient accumulation, sharded data parallelism, and efficient attention mechanisms like Flash Attention.

5. **How do you balance model deployments?**
   - Use quantization, knowledge distillation for smaller models, and model pruning to remove unnecessary parameters.

6. **What is model pruning and how does it work?**
   - Systematically removing less important weights or neurons from a model, reducing size with minimal performance impact, often followed by fine-tuning.

7. **How does batching improve inference throughput?**
   - Processing multiple inputs together utilizes hardware more efficiently, increasing throughput at the potential cost of slightly increased latency.

8. **What is continuous training in production ML systems?**
   - Regularly updating models with new data while deployed, helping maintain performance as data distributions shift over time.

9. **How do you handle concept drift in deployed models?**
   - Monitor prediction distributions, establish drift detection metrics, maintain evaluation datasets, and implement automated retraining pipelines.

10. **What are efficient serving architectures for LLMs?**
    - Multi-model servers, token streaming, compute graph optimization, hardware acceleration, and caching frequently requested outputs.

## Model Training

1. **What is curriculum learning?**
   - Training models on progressively harder examples, similar to human education, often resulting in faster convergence and better final performance.

2. **How do adaptive learning rate methods work?**
   - Algorithms like Adam or AdaGrad automatically adjust learning rates for each parameter based on historical gradients, simplifying optimization.

3. **What is transfer learning and when should it be used?**
   - Using knowledge from previously trained models on new tasks, effective when target data is limited and source and target domains share features.

4. **What techniques prevent overfitting during training?**
   - Regularization methods (L1/L2), dropout, data augmentation, early stopping, and cross-validation.

5. **How do you select appropriate hyperparameters?**
   - Use grid search for simple models, random search for efficiency, or Bayesian optimization for complex models with many hyperparameters.

6. **What is the vanishing gradient problem and how is it addressed?**
   - When gradients become extremely small in deep networks, slowing learning. Solutions include residual connections, batch normalization, and specialized activation functions like ReLU.

7. **How do activation functions affect neural network training?**
   - They introduce non-linearity enabling complex function approximation. Different activations (ReLU, GELU, Swish) affect convergence speed, gradient flow, and representation capacity.

8. **What is the role of normalization techniques in training?**
   - Techniques like batch normalization and layer normalization stabilize training by standardizing activations, enabling faster learning and reducing sensitivity to hyperparameters.

9. **How do you handle class imbalance during training?**
   - Use weighted loss functions, resampling techniques (oversampling minority classes or undersampling majority classes), data augmentation, or synthetic minority sampling.

10. **What is multi-task learning and when is it beneficial?**
    - Training a model to perform multiple related tasks simultaneously, sharing representations across tasks, often improving performance when tasks are related and data for some tasks is limited.

## Model Fine-tuning

1. **What is the difference between fine-tuning and training from scratch?**
   - Fine-tuning starts with a pre-trained model and adapts it to a specific task, requiring less data and compute than training from scratch.

2. **How does parameter-efficient fine-tuning work?**
   - Methods like LoRA, adapters, and prompt tuning update only a small subset of parameters, reducing memory requirements and preventing catastrophic forgetting.

3. **What is catastrophic forgetting and how can it be mitigated?**
   - When models lose previously learned capabilities during fine-tuning. Mitigations include regularization, knowledge distillation, or freezing portions of the model.

4. **What are the benefits of domain-specific fine-tuning?**
   - Improves performance on specialized tasks, adapts to domain vocabulary, and captures domain-specific patterns not present in general training data.

5. **What dataset quality criteria are important for fine-tuning?**
   - Task relevance, diversity, balanced representation, lack of harmful content, and accurate annotations or labels.

6. **How do you determine the optimal amount of fine-tuning data?**
   - Balance between enough data to learn task-specific patterns but not so much that the model overfits or training becomes prohibitively expensive, often determined empirically.

7. **What is instruction fine-tuning and why is it important for LLMs?**
   - Training models specifically to follow human instructions, improving their ability to perform diverse tasks from natural language prompts without task-specific fine-tuning.

8. **How do you prevent fine-tuned models from exhibiting new biases?**
   - Use balanced and representative fine-tuning data, evaluate models across demographic groups, and implement bias metrics to detect problematic patterns.

9. **What are effective evaluation strategies for fine-tuned models?**
   - Compare against baseline models, use held-out test sets, conduct human evaluations, and measure both task-specific metrics and general capabilities retention.

10. **How does few-shot learning differ from traditional fine-tuning?**
    - Few-shot learning adapts to new tasks with very limited examples (often in the prompt itself), while traditional fine-tuning modifies model weights using a larger labeled dataset.
