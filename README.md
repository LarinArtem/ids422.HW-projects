# IDS 422 — Machine Learning & Neural Networks Coursework

Assignments and a final project in deep learning and natural-language processing, progressing from neural networks implemented **from scratch in NumPy** to recurrent architectures and a PyTorch text classifier trained on Word2Vec embeddings.

## Repository contents

| Notebook | Topic |
|---|---|
| `HW1.ipynb` | Two-layer neural network built from scratch (custom `two_layer_net` module) — forward pass, backpropagation, activation-function analysis (tanh, ReLU, leaky ReLU) |
| `HW2.ipynb` | Text preprocessing and classical NLP — tokenization with NLTK, feature extraction, first text models with scikit-learn |
| `hw3_solution.ipynb` | Word embeddings — the same model implemented twice, once in **NumPy** and once in **PyTorch**, plus Gensim/Word2Vec experiments |
| `HW4.ipynb` | Recurrent neural networks — a `simple_rnn` implementation for sequence modeling |
| `HW5.ipynb` | Gated recurrence — a GRU language model (`grulm`) with training-curve visualization |
| `hw6_solution.ipynb` | Sequence-to-sequence models (`seq2seq`) — encoder–decoder architecture |
| `final_project.ipynb` | News classification on AG News with Word2Vec + PyTorch |

The notebooks rely on shared course modules (`common`, `dataset`, `trainers`, etc.) that implement layers, optimizers, and data loaders — the emphasis is on understanding each architecture's internals before using framework abstractions.

## Design of the coursework arc

The assignments deliberately climb the deep-learning stack:

1. **Foundations (HW1)** — implementing forward/backward passes manually builds intuition for gradient flow, why activation choice (tanh vs. ReLU vs. leaky ReLU) affects convergence, and where vanishing gradients come from.
2. **Text as data (HW2–HW3)** — moving from bag-of-words features to learned distributed representations (Word2Vec). HW3's dual NumPy/PyTorch implementation makes explicit what the framework automates (autograd, GPU tensors).
3. **Sequence models (HW4–HW6)** — vanilla RNN → GRU language model → encoder–decoder seq2seq, tracing exactly the architectural lineage that motivated attention/transformers.
4. **Applied capstone (final project)** — putting it together on a real dataset.

## Final project — AG News classification (`final_project.ipynb`)

Supervised text classification on the **AG News** dataset (~120,000 training / 7,600 test articles across 4 news categories):

- **Preprocessing & tokenization** with Gensim.
- **Word2Vec embeddings** trained/loaded as distributed word representations.
- **PyTorch classifier** consuming the embeddings, with `TensorDataset` + `DataLoader` for efficient mini-batch training and the **Adam** optimizer.
- **Evaluation** — accuracy tracked across epochs, with results interpreted in terms of model capacity, embedding quality, and optimization dynamics.

## Tech stack

Python, `numpy`, `torch` (PyTorch), `gensim` (Word2Vec), `nltk`, `scikit-learn`, `pandas`, `matplotlib`, plus course-provided modules (`common`, `dataset`, `trainers`).

## Running

```bash
pip install numpy torch gensim nltk scikit-learn pandas matplotlib
```
Place the course helper modules (`common/`, `dataset/`, etc.) on the Python path — several notebooks add them via `sys.path` — then run each notebook top to bottom. The final project downloads/expects the AG News dataset.
