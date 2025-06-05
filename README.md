# 🧠 Wordle-RL: Learning to Play Wordle Using Rewards

This project explores how an agent can learn to play the game of Wordle using reinforcement learning (REINFORCE with policy gradients) instead of traditional supervised learning using minimizing of loss functions via gradient decent.

---

## 🌍 Environment

A lightweight Wordle environment is implemented from scratch:

- **Vocabulary**: Real 5-letter English words using `nltk.corpus.words`.
- **Game Logic**:
  - Up to **6 turns** per game.
  - Feedback system mimics Wordle using:
    - ⬛ (absent),
    - 🟨 (present),
    - 🟩 (correct).
- **Reward Design**:
  - +0.1 for each 🟩
  - +0.05 for each 🟨
  - +1 / (1 + turns taken) if the correct word is guessed
  - +1 bonus for successful guess (binary success)

---

## 🤖 Policy

The agent is a GRU-based RNN trained with the REINFORCE algorithm:

- **Inputs**: A sequence of (guess, feedback) tokenized pairs.
- **Model**:
  - Embedding layer
  - GRU (hidden_dim = 64)
  - Linear layer projecting to vocabulary space
- **Output**: A probability distribution over possible 5-letter words

Learning is done by maximizing expected reward:

The model learns purely from interaction — no ground-truth labels or supervised loss.

---

## 📊 Performance

After 50,000 training episodes:

- ✅ **Success Rate**: 98%
- 🎯 **Average Turns (Successful Games)**: 2.59
- 🧮 **Average Reward**: 1.87

---

## 🛠️ Tech Stack

- Python, PyTorch
- NLTK for English vocabulary
- Custom Wordle RL Environment
- Policy Gradient (REINFORCE)

---

## 🧪 How to Run
- use the jupyter notebook provided

---

## 🧩 Open Problems

- ❗ **Scaling to Large Vocabulary**: The model struggles to converge when trained on a large vocabulary set, likely due to increased action space complexity and sparse reward signals.
- ❗ **Delayed Rewards**: If the rewards are given only at the end of the game, training is slow and unstable in early episodes.
- ❗ **Fixed Vocabulary Distribution**: The model explicitly learns a probability distribution over a fixed vocabulary, unlike language models which generalize and adapt via weight updates to in-context tasks. This limits flexibility and sample efficiency.

---

## 🔭 Next Steps

- 🔍 **Improve Convergence on Large Vocabularies**: Explore curriculum learning, better initialization, entropy regularization or hierarchical action spaces to improve stability with larger vocab sizes.
- 🧪 **Directly Fine-Tune an LLM**: Investigate how a pretrained LLM (e.g., GPT-style) could be fine-tuned or prompted to adapt to Wordle-style tasks using policy gradient.
- 🎯 **Explore Intermediate Rewards**: Add more granular reward shaping to encourage learning partial structure (e.g., rewarding information gain, entropy reduction).
- 🧠 **Compare Against Supervised and Search-Based Agents**: Benchmark this reinforcement learning approach against supervised learning baselines and tree-search-based agents (e.g., minimax-style guessers with entropy-based heuristics).

---

