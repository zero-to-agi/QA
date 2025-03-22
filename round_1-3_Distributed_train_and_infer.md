
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


**1. What is DeepSpeed ZeRO-Infinity and how does it expand model scale beyond GPU memory limitations?
**- DeepSpeed ZeRO-Infinity extends ZeRO-3 by introducing offloading techniques that utilize both CPU RAM and NVMe storage, effectively creating a virtually unlimited memory pool. It employs strategic partitioning, offloading, and prefetching policies to minimize performance impact while enabling training of trillion-parameter models on limited GPU resources. ZeRO-Infinity also optimizes communication patterns and implements smart memory management to reduce data movement between different memory tiers.

**2. How do quantization techniques impact distributed inference performance?**
- Quantization reduces model precision (e.g., from FP32 to INT8 or INT4) to decrease memory footprint and computational requirements, enabling faster inference and reduced communication overhead in distributed settings. While it can introduce minor accuracy degradation, techniques like quantization-aware training, post-training quantization, and mixed-precision inference help maintain model quality. In distributed systems, quantization can significantly improve throughput, reduce latency, and allow larger batch sizes or model shards per device.

**3. What are activation checkpointing and recomputation, and why are they crucial for large-scale distributed training?**
- Activation checkpointing (or gradient checkpointing) reduces memory usage by saving only select activations during the forward pass and recomputing the others during backpropagation. This trades additional computation for dramatically reduced memory requirements, enabling training of deeper models that would otherwise exceed available memory. In distributed settings, this technique allows fitting larger model shards per device and can be combined with parallelism strategies to optimize the memory-computation tradeoff.

**4. How does distributed pretraining differ from distributed fine-tuning in terms of strategy and resource requirements?**
- Pretraining typically requires massive datasets, longer training durations, and more extensive computational resources, often employing complex parallelism strategies across many devices. Fine-tuning is more lightweight, often achievable with data parallelism alone, and requires fewer resources. The communication patterns and optimization techniques also differ; pretraining benefits from techniques like ZeRO and 3D parallelism, while fine-tuning may prioritize techniques like LoRA or parameter-efficient tuning methods that reduce distributed overhead.

**5. What is sequence parallelism and how does it complement other distributed training strategies?**
- Sequence parallelism splits sequences within a batch across multiple devices, reducing the per-device memory needed for storing activations in attention layers. This specifically targets the sequence length dimension, which is particularly memory-intensive in transformer architectures. It complements tensor parallelism (which splits model parameters) by providing additional memory savings for long sequences. This enables training with longer context windows and integrates naturally within 3D/4D parallelism frameworks.

**6. How do heterogeneous network conditions affect distributed training strategies and performance?**
- Network heterogeneity (varying bandwidth, latency, and reliability between nodes) can create stragglers that slow down synchronous training. Adaptive strategies include using hierarchical communication topologies that optimize for network topology, gradient compression techniques that reduce communication volume, asynchronous or semi-synchronous update schemes that mitigate straggler effects, and network-aware scheduling that places heavily-communicating processes on nodes with better connectivity.

**7. What are the challenges and solutions for distributed reinforcement learning compared to supervised learning?**
- Distributed RL faces unique challenges including temporal dependency in data, exploration-exploitation tradeoffs, and often unstable learning dynamics. Solutions include asynchronous methods like A3C that run multiple environments in parallel, distributed experience replay that gathers experiences from many actors, value decomposition for multi-agent settings, and hybrid architectures that combine centralized training with distributed execution. These approaches must handle both environment parallelism and model parallelism considerations.

**8. How do frameworks like Ray, Horovod, and PyTorch Distributed compare for distributed training implementations?**
- Ray offers a flexible, Python-native distributed computing framework with libraries like Ray Train for distributed ML and dynamic resource scaling. Horovod provides a lightweight ring-allreduce implementation that integrates with multiple frameworks and excels at pure data parallelism. PyTorch Distributed offers tight framework integration with comprehensive support for all parallelism strategies (FSDP, DDP, etc.) but may require more configuration. The choice depends on factors like existing infrastructure, parallelism needs, and framework preferences.
