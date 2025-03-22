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

## Distributed Training & Inference

1. **What is data parallelism in distributed training?**
   - Distributing batches of data across multiple devices, each computing gradients independently, then synchronizing updates—the most common form of parallelism.

2. **How does model parallelism work?**
   - Splitting a model's layers or parameters across multiple devices, allowing models too large for a single device to be trained or run, at the cost of communication overhead.

3. **What is pipeline parallelism?**
   - Dividing model layers into stages across devices, with each device processing different stages of the forward and backward passes, balancing computation and communication.

4. **What are different synchronization strategies for distributed SGD?**
   - Synchronous (wait for all workers to compute gradients), asynchronous (update immediately when gradients arrive), and semi-synchronous approaches (wait for a subset of workers).

5. **What communication patterns are used in distributed training?**
   - All-reduce (each device gets complete gradients), parameter server (central server aggregates updates), and ring-allreduce (efficient communication pattern minimizing bandwidth).

6. **How do you handle device failures in distributed training?**
   - Implement checkpointing, elastic scaling (continue with fewer devices), gradient accumulation to handle temporary failures, and automatic restart capabilities.

7. **What is Zero Redundancy Optimizer (ZeRO)?**
   - A memory optimization technique that shards model parameters, gradients, and optimizer states across devices to enable training larger models without full model replication.

8. **How does distributed inference differ from distributed training?**
   - Inference focuses on throughput and latency without gradient computation, often using different parallelization strategies and specialized serving infrastructures.
   - While training focuses on throughput with static shapes, distributed inference requires low latency, handling dynamic workloads, and managing concurrent requests with varying sequence lengths. This fundamental difference necessitates specialized optimization strategies for each. This distinction is critical as deployment considerations are often overlooked but are essential for real-world AI applications.

9. **What are the challenges of scaling batch size in distributed training?**
   - Large batch sizes can degrade generalization performance, require learning rate adjustments, and may hit diminishing returns in training efficiency.

10. **How do heterogeneous computing resources affect distributed training strategies?**
    - Require load balancing, adaptation to different device capabilities, specialized communication patterns, and potentially asymmetric data or model partitioning.
    - 
11. **What is Fully Sharded Data Parallelism (FSDP) and how does it improve distributed training?**
   - FSDP shards model parameters, gradients, and optimizer states across all devices, significantly reducing memory requirements compared to traditional data parallelism. It enables training larger models with fewer GPUs by eliminating parameter redundancy across devices, making it one of the most important recent advancements in efficient large-scale model training.
   - 
12. **What is 3D parallelism in distributed training?**
   - 3D parallelism combines data, tensor, and pipeline parallelism strategies in a single training system. This approach allows for scaling along multiple dimensions simultaneously, optimizing for different hardware configurations and model architectures. It represents the state-of-the-art approach for training truly massive models that wouldn't be possible with any single parallelism strategy alone.

13. **How does vLLM optimize distributed inference for large language models?**
   - vLLM uses PagedAttention, inspired by operating system virtual memory design, to optimize GPU memory management during inference. This technique improves serving performance up to 24x by efficiently handling attention key-value caches and enabling higher throughput with lower latency. This represents one of the most significant breakthroughs in making large language models practically deployable at scale.

14. **What are the key differences between distributed training and distributed inference requirements?**
   - While training focuses on throughput with static shapes, distributed inference requires low latency, handling dynamic workloads, and managing concurrent requests with varying sequence lengths. This fundamental difference necessitates specialized optimization strategies for each[^7](https://blog.vllm.ai/2025/02/17/distributed-inference.html).

15. **How does tensor parallelism differ from other parallelism strategies?**
   - Tensor parallelism splits individual tensors and their operations across multiple devices, specifically targeting computationally intensive operations like large matrix multiplications. This approach is particularly effective for scaling model width and maintaining computational efficiency with very large hidden dimensions[^1](https://www.run.ai/blog/parallelism-strategies-for-distributed-training).

16. **What is SimpleFSDP and how does it improve on standard FSDP?**
   - SimpleFSDP is a PyTorch-native compiler-based implementation of Fully Sharded Data Parallel that simplifies maintenance and debugging while integrating with torch.compile for improved performance. It provides a more streamlined approach to implementing sharded data parallelism[^6](https://arxiv.org/html/2411.00284v1).

17. **What communication patterns are used in distributed inference systems?**
   - Distributed inference systems employ various communication patterns including tensor-parallel communication for splitting model layers, pipeline-parallel communication for sequential processing, and request-level load balancing for distributing inference requests across multiple servers[^9](https://docs.vllm.ai/en/latest/serving/distributed_serving.html).

18. **How do different parallelism strategies affect memory usage versus computation time?**
   - Data parallelism typically maintains memory usage but reduces computation time linearly with devices. Model/tensor parallelism reduces memory per device but adds communication overhead. Pipeline parallelism balances memory and computation but introduces pipeline bubbles. The 
optimal strategy depends on model size, hardware configuration, and performance requirements. Understanding these tradeoffs is crucial for designing efficient distributed systems.

19. **What challenges arise when scaling FSDP to very large models?**
   - Challenges include communication overhead from frequent parameter resharding, overlapping computation with communication, managing optimization state across devices, handling mixed precision arithmetic, and implementing efficient checkpointing strategies[^5](https://www.vldb.org/pvldb/vol16/p3848-huang.pdf).

20. **How are dynamic batch sizes handled in distributed inference systems?**
   - Distributed inference systems handle dynamic batch sizes through continuous batching techniques, which process requests as they arrive without waiting for fixed batch boundaries. This approach, combined with efficient memory management like PagedAttention, allows for maximizing throughput while maintaining low latency, even with varying sequence lengths[^7](https://blog.vllm.ai/2025/02/17/distributed-inference.html).

21. **What is elastic training in distributed systems?**
   - Elastic training allows distributed training jobs to dynamically add or remove computing resources during execution without restarting the entire job. This provides fault tolerance against hardware failures and enables flexible resource allocation in shared computing environments.

22. **How does heterogeneous GPU training work in modern distributed systems?**
   - Heterogeneous GPU training adapts workloads to different GPU capabilities within the same training job, assigning appropriate portions of models or data based on memory capacity and computational power of each device. Modern frameworks implement load balancing strategies to maximize resource utilization across diverse hardware.


# Additional Q&As on Distributed Training & Inference

## 1. What is DeepSpeed ZeRO-Infinity and how does it expand model scale beyond GPU memory limitations?
- DeepSpeed ZeRO-Infinity extends ZeRO-3 by introducing offloading techniques that utilize both CPU RAM and NVMe storage, effectively creating a virtually unlimited memory pool. It employs strategic partitioning, offloading, and prefetching policies to minimize performance impact while enabling training of trillion-parameter models on limited GPU resources. ZeRO-Infinity also optimizes communication patterns and implements smart memory management to reduce data movement between different memory tiers.

## 2. How do quantization techniques impact distributed inference performance?
- Quantization reduces model precision (e.g., from FP32 to INT8 or INT4) to decrease memory footprint and computational requirements, enabling faster inference and reduced communication overhead in distributed settings. While it can introduce minor accuracy degradation, techniques like quantization-aware training, post-training quantization, and mixed-precision inference help maintain model quality. In distributed systems, quantization can significantly improve throughput, reduce latency, and allow larger batch sizes or model shards per device.

## 3. What are activation checkpointing and recomputation, and why are they crucial for large-scale distributed training?
- Activation checkpointing (or gradient checkpointing) reduces memory usage by saving only select activations during the forward pass and recomputing the others during backpropagation. This trades additional computation for dramatically reduced memory requirements, enabling training of deeper models that would otherwise exceed available memory. In distributed settings, this technique allows fitting larger model shards per device and can be combined with parallelism strategies to optimize the memory-computation tradeoff.

## 4. How does distributed pretraining differ from distributed fine-tuning in terms of strategy and resource requirements?
- Pretraining typically requires massive datasets, longer training durations, and more extensive computational resources, often employing complex parallelism strategies across many devices. Fine-tuning is more lightweight, often achievable with data parallelism alone, and requires fewer resources. The communication patterns and optimization techniques also differ; pretraining benefits from techniques like ZeRO and 3D parallelism, while fine-tuning may prioritize techniques like LoRA or parameter-efficient tuning methods that reduce distributed overhead.

## 5. What is sequence parallelism and how does it complement other distributed training strategies?
- Sequence parallelism splits sequences within a batch across multiple devices, reducing the per-device memory needed for storing activations in attention layers. This specifically targets the sequence length dimension, which is particularly memory-intensive in transformer architectures. It complements tensor parallelism (which splits model parameters) by providing additional memory savings for long sequences. This enables training with longer context windows and integrates naturally within 3D/4D parallelism frameworks.

## 6. How do heterogeneous network conditions affect distributed training strategies and performance?
- Network heterogeneity (varying bandwidth, latency, and reliability between nodes) can create stragglers that slow down synchronous training. Adaptive strategies include using hierarchical communication topologies that optimize for network topology, gradient compression techniques that reduce communication volume, asynchronous or semi-synchronous update schemes that mitigate straggler effects, and network-aware scheduling that places heavily-communicating processes on nodes with better connectivity.

## 7. What are the challenges and solutions for distributed reinforcement learning compared to supervised learning?
- Distributed RL faces unique challenges including temporal dependency in data, exploration-exploitation tradeoffs, and often unstable learning dynamics. Solutions include asynchronous methods like A3C that run multiple environments in parallel, distributed experience replay that gathers experiences from many actors, value decomposition for multi-agent settings, and hybrid architectures that combine centralized training with distributed execution. These approaches must handle both environment parallelism and model parallelism considerations.

## 8. How do frameworks like Ray, Horovod, and PyTorch Distributed compare for distributed training implementations?
- Ray offers a flexible, Python-native distributed computing framework with libraries like Ray Train for distributed ML and dynamic resource scaling. Horovod provides a lightweight ring-allreduce implementation that integrates with multiple frameworks and excels at pure data parallelism. PyTorch Distributed offers tight framework integration with comprehensive support for all parallelism strategies (FSDP, DDP, etc.) but may require more configuration. The choice depends on factors like existing infrastructure, parallelism needs, and framework preferences.

Would you like me to focus on any particular aspect of distributed training and inference for more detailed Q&As?

=========================================================================

AI & Machine Learning: Questions & Answers
Domain Expertise - Artificial Intelligence




How has the AI alignment problem changed over time?
The AI alignment problem involves making sure that AI models match human intentions and goals. Earlier, this was mostly a theoretical topic; now, it's become more practical, and new techniques (like Reward Learning from Human Feedback or RLHF) help address it.

What is explainable AI (XAI)?
Explainable AI means developing AI systems that clearly show why and how they made certain decisions. This helps humans understand, trust, and control AI more effectively.

How does federated learning differ from traditional centralized learning?
In traditional centralized learning, you bring all data to one place to train a model. Federated learning keeps data on individual devices (like phones), trains smaller models locally on each device, and then combines these models. This approach helps protect privacy and reduces data sharing.

What are the main ways AI represents knowledge?
AI systems represent knowledge in different ways, such as through rules, logic statements, semantic networks (connected concepts), frames (structured templates), and more recently using embeddings and knowledge graphs to link concepts together.

What is the role of heuristics in AI problem-solving?
Heuristics are simple guidelines ("rules of thumb") that help AI find solutions quickly, without needing to thoroughly check every possibility. While they sometimes don't find the perfect solution, they save time and effort by giving good, practical answers fast.

How does multi-agent AI differ from single-agent systems?
Single-agent AI involves just one agent making decisions by itself. In multi-agent AI, several independent agents interact, cooperate, or compete with each other. This often involves coordination, conflict handling, and principles from game theory.

Generative AI
How do diffusion models make images?
Diffusion models start by adding random noise to images to destroy them, and then they slowly learn to reverse this process. Once trained, these models can take random noisy images and gradually remove that noise to create new clear images.

What is the main idea behind transformer models used in generative AI?
Transformer models use something called self-attention, which helps them understand which parts of the input data are most important. This lets them find links in words or ideas even over long distances, improving their ability to create meaningful output.

What are common sampling methods used when AI generates text?
Some popular methods include taking the most likely next word ("greedy decoding"), exploring multiple possible words ("beam search"), adjusting randomness ("temperature sampling"), and picking from just the best options ("nucleus" or "top-p" sampling).

How do diffusion models compare with GANs?
GANs (Generative Adversarial Networks) use two models competing with each other but can often be tricky to train. Diffusion models use a simpler, step-by-step noise-removal process. They tend to produce better quality images with more variety and are often easier to train.

What improvements led from GPT-2 to GPT-3 models?
GPT-3 is much larger, having many more "neurons" or parameters (175 billion versus GPT-2's 1.5 billion). GPT-3 also used more advanced training methods and higher-quality data, allowing it to handle tasks better and even gain new abilities.

What is controllable generation in generative AI?
Controllable generation means you can precisely influence certain characteristics of what the model creates. For example, you might control the tone, style, sentiment, or content of messages by specifying clearly what you want the model to produce.

How do models like DALL-E create images from text?
DALL-E and similar models first convert images into smaller "tokens" or pieces, kind of like words. They then use language models to generate these tokens based on a text description and finally turn these tokens back into images.

What is the role of latent spaces in generative models?
Latent spaces are simplified "maps" or representations of concepts where similar ideas naturally group together. Using latent spaces helps AI smoothly generate new outputs, explore different variations, or modify generated content easily.

How do AI systems generate music, like MusicLM?
These models usually turn audio (music) into sequences of tokens or visual representations (spectrograms). They learn patterns from these sequences, and afterward, they create new sequences based on what they've learned and then convert these sequences back into musical audio.

What are common challenges when evaluating generative AI models?
Evaluating generative AI can be difficult because the output is subjective. Often there's no clear measure of "correctness," making it tricky to know if one model is better than another. Challenges include judging creativity, diversity, consistency, and the quality of the content created.
