Training the Prover and Verifier Models

Both helpful provers (producing correct solutions) and sneaky provers (producing incorrect solutions that fool the verifier) are iteratively trained against a verifier. This training process results in improved prover accuracy and verifier robustness. The specific method of training is described in source as a version of the prover-verifier game initially proposed by Anil et al.
●
The verifier is trained first, using a balanced mix of samples from all provers from previous training rounds.
●
The prover is then trained using reinforcement learning (RL), with a reward system based on two factors:
○
Whether the solution is correct, based on the prover's assigned role (helpful or sneaky).
○
How high the verifier score is for the solution, again depending on the assigned role.
So, while RL is a critical part of the training process, the training methodology is more complex than just RLHF (Reinforcement Learning from Human Feedback). This process uses smaller verifier LLMs as proxies for human judgment and involves a competitive training dynamic between the provers and the verifier. The process could be summarized as follows:
1.
Verifier training: The verifier model is trained first to predict the correctness of solutions. It learns from a balanced mixture of solutions from previous training rounds, including correct solutions from helpful provers and incorrect but convincing solutions from sneaky provers.
2.
Prover Training: The prover model is trained using reinforcement learning to generate solutions that satisfy its assigned role and convince the verifier. The reward for the prover is a combination of:
○
Role Alignment: Whether the solution is correct (for helpful provers) or incorrect (for sneaky provers).
○
Convincingness: How high a score the verifier assigns to the solution.
The combination of these two rewards encourages the provers to generate solutions that are both role-aligned and convincing to the verifier. The iterative training process helps both the prover and the verifier to improve their performance over time.
The sources also note the importance of:
●
Reward function: The specific reward function used to train the prover significantly impacts the training dynamics and the resulting model performance.
●
Verifier size: The relative size of the verifier compared to the prover matters for the effectiveness of the training. Too small or too large verifiers can lead to suboptimal results.
Overall, this method aims to strike a balance between model performance and legibility by training the model to produce solutions that are both accurate and easily checkable by a smaller verifier model, which serves as a proxy for human judgment.


Let me analyze potential blockers specifically in the context of implementing Prover-Verifier games for your sandwiching evaluation research:

1. Model Capability/Size Constraints:
- Your current model setup (Gemma-2 2B-it, Qwen2-1.5B-Instruct, Llama-3.1-8B-Instruct) may be too small compared to those used in the original paper
- Particularly problematic for training verifiers, as they need to be robust enough to evaluate solutions
- The gap between your models might be too large for effective prover-verifier dynamics

2. Training Dynamics Issues:
- Your setup focuses on direct supervision (R=0/R=1 quality judgments) while Prover-Verifier games require more complex training dynamics
- The iterative training process (verifier training followed by prover RL) would need to be maintained across different expertise levels
- It's unclear how to maintain the "sneaky prover" concept across different model capability levels

3. Experimental Design Conflicts:
- Your current setup tests simple supervision across layers, while Prover-Verifier requires adversarial training
- The "sandwiching" evaluation might become more complex when both provers and verifiers exist at each layer
- Need to determine whether to evaluate:
  * The prover's performance
  * The verifier's accuracy
  * Or some combination of both

4. Resource Intensive:
- Training both provers and verifiers at each layer requires significantly more computational resources
- Need multiple training rounds for each layer to achieve the competitive dynamic
- May be impractical with your current experimental scope

5. Evaluation Metrics Mismatch:
- Your current metrics focus on straightforward accuracy and performance gaps
- Prover-Verifier games introduce additional dimensions like:
  * Solution legibility
  * Verifier robustness
  * Prover convincingness
- Need to determine how these align with your research questions about scalable oversight

6. Layer Definition Complexity:
- Current setup has clear layer definitions based on model size and token limits
- Prover-Verifier would require defining:
  * What constitutes a "helpful" vs "sneaky" prover at each layer
  * How verifier capabilities should scale across layers
  * Whether verifiers should be smaller than provers at each layer

7. Ground Truth Usage:
- Your setup uses ground truth as the highest layer
- In Prover-Verifier games, ground truth is used throughout training to determine solution correctness
- Need to clarify how this affects the sandwiching evaluation structure

8. Scalability Questions:
- Your research aims to understand how oversight methods scale
- Prover-Verifier games might introduce additional scaling challenges:
  * How does the adversarial dynamic change across expertise levels?
  * Do "sneaky" behaviors transfer across layers?
  * Does legibility improvement scale with model capability?

These blockers suggest that implementing Prover-Verifier games in your sandwiching evaluation framework would require significant modifications to either:
1. The Prover-Verifier methodology to fit your experimental constraints
2. Your experimental design to accommodate the full Prover-Verifier training dynamics
