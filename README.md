# 🤖 NovaGPT

### A GPT-Style Language Model Built From Scratch in PyTorch

NovaGPT is a small **decoder-only, GPT-style language model implemented from scratch using PyTorch**. The project was built to understand the internal mechanics of Transformer-based language models rather than simply using a pretrained model or an API.

The project covers the complete lifecycle of a language model:

> **Raw Text → Tokenization → Embeddings → Transformer → Training → Evaluation → Text Generation → Inference → Web Application → Cloud Deployment**

---

## 🚀 Live Demo

🔗 **Hugging Face Space:**
**https://huggingface.co/spaces/dhairya-105/NovaGPT**

The deployed application provides an interactive browser interface where users can enter a prompt and generate text using the trained model.

---

## 📌 Project Overview

NovaGPT is an **autoregressive character-level language model** based on the GPT/Transformer architecture.

The model was implemented from the ground up using PyTorch, including:

* Character-level tokenization
* Token embeddings
* Positional embeddings
* Causal self-attention
* Multi-head attention
* Transformer blocks
* Residual connections
* Layer normalization
* Feed-forward neural networks
* GELU activation
* Dropout
* Cross-entropy loss
* Backpropagation
* AdamW optimization
* Gradient clipping
* Cosine learning-rate scheduling
* Validation and checkpointing
* Temperature sampling
* Top-K sampling
* CUDA GPU training
* Gradio inference interface
* Hugging Face Spaces deployment

---

# 🏗️ Architecture

NovaGPT follows a **decoder-only Transformer architecture**, similar in principle to GPT-style language models.

```text
                    Input Text
                        │
                        ▼
              Character Tokenization
                        │
                        ▼
                 Token Embeddings
                        +
              Positional Embeddings
                        │
                        ▼
              ┌─────────────────────┐
              │   Transformer Block │
              │                     │
              │ Multi-Head Attention│
              │        ↓            │
              │ Residual + LayerNorm│
              │        ↓            │
              │ Feed-Forward + GELU │
              │        ↓            │
              │ Residual + LayerNorm│
              └─────────────────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │   Transformer Block │
              └─────────────────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │   Transformer Block │
              └─────────────────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │   Transformer Block │
              └─────────────────────┘
                        │
                        ▼
                Final Linear Layer
                        │
                        ▼
                 Vocabulary Logits
                        │
                        ▼
              Temperature / Top-K
                        │
                        ▼
                 Next Character
                        │
                        ▼
                 Generated Text
```

---

# ⚙️ Model Configuration

| Component              |   Configuration |
| ---------------------- | --------------: |
| Vocabulary Size        |              65 |
| Tokenization           | Character-level |
| Embedding Dimension    |             128 |
| Transformer Layers     |               4 |
| Attention Heads        |               4 |
| Feed-Forward Expansion |              4× |
| Context Length         |             128 |
| Dropout                |             0.1 |
| Parameters             |           ~826K |

---

# 🔤 Tokenization

NovaGPT uses **character-level tokenization**.

Every unique character in the dataset is assigned an integer ID.

For example:

```text
"a"  →  token ID
"b"  →  token ID
" "  →  token ID
"\n" →  token ID
```

Two mappings are used:

```python
stoi
```

Maps characters to integer token IDs.

```python
itos
```

Maps integer token IDs back to characters.

The tokenized text is then converted into PyTorch tensors for training.

### Why character-level tokenization?

Character-level tokenization was intentionally chosen because it is simple and makes the mechanics of language modeling easier to understand.

However, it has an important limitation: the model has to learn words, spelling, punctuation, and language structure from individual characters.

A future version would use a **subword tokenizer such as BPE**.

---

# 🧠 Embeddings

After tokenization, each token ID is converted into a dense vector using a learned embedding layer.

The model uses:

```text
embedding_dim = 128
```

Therefore, every token is represented by a vector containing **128 learned numerical values**.

The model also uses learned positional embeddings.

The token and positional embeddings are added together:

```text
Token Embedding
       +
Position Embedding
       ↓
Transformer Input
```

This provides the model with information about both:

* **What the token is**
* **Where the token occurs**

---

# 👁️ Self-Attention

Self-attention allows each token to examine other tokens in its context and determine which ones are relevant.

The attention mechanism creates:

* **Query (Q)**
* **Key (K)**
* **Value (V)**

The core attention equation is:

```text
Attention(Q, K, V)
=
softmax(QKᵀ / √dₖ)V
```

The query and key vectors determine how relevant tokens are to one another, while the value vectors contain the information that is combined to create the attention output.

---

# 🔒 Causal Self-Attention

Because NovaGPT is an autoregressive GPT-style model, it uses **causal attention**.

A token is not allowed to see future tokens.

For example:

```text
The king went to
```

When predicting the next character, the model can only use information that appears before the prediction position.

This is implemented using a **causal attention mask**.

```text
Past tokens      → Visible
Current token    → Visible
Future tokens    → Hidden
```

This prevents information leakage during training.

---

# 🧩 Multi-Head Attention

NovaGPT uses:

```text
4 attention heads
```

Instead of using a single attention mechanism, multiple attention heads process the representation in parallel.

Different heads can learn different relationships between tokens.

The resulting attention outputs are combined and projected back into the model's embedding dimension.

---

# 🔄 Transformer Block

Each Transformer block contains:

1. Multi-Head Causal Self-Attention
2. Dropout
3. Residual Connection
4. Layer Normalization
5. Feed-Forward Network
6. GELU Activation
7. Another Residual Connection
8. Another Layer Normalization

The structure is:

```text
Input
  │
  ▼
Multi-Head Self-Attention
  │
  ▼
Dropout
  │
  ▼
Residual + LayerNorm
  │
  ▼
Feed-Forward Network
  │
  ▼
GELU
  │
  ▼
Dropout
  │
  ▼
Residual + LayerNorm
  │
  ▼
Output
```

NovaGPT contains **4 Transformer blocks**.

---

# ⚡ Feed-Forward Network

After attention, each token representation passes through a feed-forward neural network.

The model uses:

```text
128
 ↓
512
 ↓
GELU
 ↓
128
```

The intermediate dimension is:

```text
128 × 4 = 512
```

The network expands the representation, applies a nonlinear GELU activation, and then projects it back to the original embedding dimension.

---

# 🔗 Residual Connections

Residual connections allow information to flow directly through the network.

Conceptually:

```python
output = x + sublayer_output
```

They help preserve information and improve gradient flow during training.

---

# 📏 Layer Normalization

Layer Normalization is applied around the attention and feed-forward components.

It helps stabilize the activations and makes Transformer training more reliable.

---

# 🎲 Dropout & Regularization

NovaGPT uses:

```text
dropout = 0.1
```

Dropout randomly removes a small portion of activations during training.

This acts as **regularization** and helps reduce overfitting.

---

# 🎯 Training Objective

NovaGPT is trained using **next-character prediction**.

The model receives a sequence of characters and learns to predict the next character at every position.

Conceptually:

```text
Input:
The king wen

Target:
he king went...
```

More generally:

```text
Current Character Sequence
            ↓
      Transformer
            ↓
    Next Character
```

The process is repeated across the entire training sequence.

---

# 📉 Loss Function

The model uses **Cross-Entropy Loss**.

At every position, the model produces a probability distribution over the vocabulary.

The loss measures how well that distribution predicts the actual next character.

Training follows:

```text
Input
  ↓
Prediction
  ↓
Cross-Entropy Loss
  ↓
Backpropagation
  ↓
Gradients
  ↓
AdamW
  ↓
Updated Parameters
```

A high loss means the model assigned relatively low probability to the correct target.

A low loss means the model assigned higher probability to the correct target.

---

# 🔧 Optimizer

NovaGPT uses:

```text
AdamW
```

AdamW is an adaptive optimization algorithm that updates model parameters based on gradient information.

It also provides **decoupled weight decay**, which acts as a form of regularization.

---

# ✂️ Gradient Clipping

Gradients are clipped using:

```text
max_norm = 1.0
```

Gradient clipping prevents extremely large gradients from destabilizing training.

---

# 📈 Learning Rate Scheduling

The project uses a **Cosine Annealing Learning Rate Scheduler**.

The learning rate gradually decreases throughout training.

This allows larger parameter updates earlier in training and progressively smaller updates later.

---

# 💻 GPU Training

Training was performed using:

### NVIDIA GeForce RTX 3050 Laptop GPU

The model uses CUDA through PyTorch for GPU acceleration.

Device selection:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

Initially, the training environment was using the CPU.

After configuring a CUDA-enabled PyTorch environment, training was moved to the NVIDIA RTX 3050.

---

# 📚 Dataset

The model was trained using the **Tiny Shakespeare** dataset.

The dataset consists of Shakespearean text and is commonly used for experimenting with small character-level language models.

Its relatively small size makes it practical for training a Transformer model on consumer hardware.

---

# 🧪 Training & Validation

The dataset was divided into training and validation subsets.

The training process monitored:

* Training Loss
* Validation Loss
* Perplexity
* Learning Rate

A checkpoint was saved whenever the validation loss improved.

The best checkpoint was stored as:

```text
checkpoints/best_model.pth
```

---

# 📊 Training Results

An early checkpoint achieved approximately:

```text
Training Loss:    1.4498
Validation Loss:  1.4864
Perplexity:       4.4210
```

After longer training, the final epoch reached approximately:

```text
Training Loss:    0.9712
Validation Loss:  1.6942
Perplexity:       5.4423
```

The increasing validation loss while training loss continued to decrease indicated **overfitting**.

Therefore, the best validation checkpoint was selected for inference rather than automatically using the final epoch.

---

# ⚠️ Overfitting

The project demonstrated a classic overfitting pattern:

```text
Training Loss
     ↓↓↓↓↓
     0.97

Validation Loss
     ↑
     1.69
```

This means the model became increasingly good at fitting the training data while becoming worse at generalizing to unseen validation data.

The project addressed this by:

* Monitoring validation loss
* Using dropout
* Using AdamW weight decay
* Saving the best validation checkpoint

---

# ✍️ Text Generation

NovaGPT generates text autoregressively.

The generation process is:

```text
Prompt
  ↓
Predict probability distribution
  ↓
Sample next character
  ↓
Append character
  ↓
Predict again
  ↓
Repeat
```

Generation continues until the requested number of tokens has been produced.

---

# 🌡️ Temperature Sampling

Temperature controls the randomness of generation.

### Lower temperature

```text
More predictable
More conservative
```

### Higher temperature

```text
More random
More diverse
```

The deployed demo uses a default temperature around:

```text
0.7
```

---

# 🔝 Top-K Sampling

Top-K sampling restricts the candidate tokens to the K most probable tokens.

The deployed demo commonly uses:

```text
top_k = 20
```

This prevents extremely unlikely tokens from being sampled while still allowing some diversity.

---

# 🌐 Web Application

The trained model was separated from the command-line generation script using:

```text
inference.py
```

The inference module is responsible for:

1. Loading the vocabulary
2. Reconstructing the model
3. Loading the trained weights
4. Selecting CPU/GPU execution
5. Encoding prompts
6. Generating tokens
7. Decoding tokens back into text

The web interface is implemented in:

```text
app.py
```

using **Gradio**.

The interface provides controls for:

* Prompt
* Temperature
* Top-K
* Maximum generated tokens

---

# ☁️ Deployment

NovaGPT was deployed using:

* **Hugging Face Spaces**
* **Gradio**
* **Hugging Face ZeroGPU**

The Space uses:

```text
app.py
```

as its application entry point.

The trained checkpoint is loaded from:

```text
checkpoints/best_model.pth
```

For ZeroGPU execution, the generation function uses:

```python
@spaces.GPU
```

This allows the inference function to request GPU resources when generating text.

---

# 🗂️ Project Structure

```text
NovaGPT/
│
├── app.py
├── inference.py
├── config.py
├── train.py
├── generate.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── dataset.py
│   └── tiny_shakespeare.txt
│
├── models/
│   ├── gpt.py
│   ├── transformer_block.py
│   ├── multi_head_attention.py
│   ├── self_attention.py
│   └── __init__.py
│
└── checkpoints/
    └── best_model.pth
```

---

# 🛠️ Tech Stack

| Category                | Technology                     |
| ----------------------- | ------------------------------ |
| Programming Language    | Python                         |
| Deep Learning Framework | PyTorch                        |
| GPU Acceleration        | CUDA                           |
| GPU                     | NVIDIA RTX 3050 6GB            |
| Model Architecture      | Decoder-only Transformer / GPT |
| Tokenization            | Character-level                |
| Dataset                 | Tiny Shakespeare               |
| Activation              | GELU                           |
| Normalization           | LayerNorm                      |
| Optimizer               | AdamW                          |
| Loss                    | Cross-Entropy                  |
| Scheduler               | Cosine Annealing               |
| Regularization          | Dropout + Weight Decay         |
| Interface               | Gradio                         |
| Deployment              | Hugging Face Spaces            |
| GPU Deployment          | Hugging Face ZeroGPU           |

---

# 📈 What the Model Learned

The trained model successfully learned recognizable patterns from the Shakespeare dataset, including:

* English character sequences
* Word formation
* Punctuation
* Capitalization
* Dialogue formatting
* Shakespeare-style vocabulary
* Character names
* Local grammatical patterns

Example generations showed recognizable structures such as:

```text
QUEEN MARGARET:
...
RICHARD:
...
CORIOLANUS:
...
```

The model does not consistently produce long, semantically coherent passages, which is expected given its small size, character-level tokenizer, short context window, and limited training dataset.

---

# ⚠️ Limitations

NovaGPT is an **educational and experimental language model**, not a production-scale foundation model.

Current limitations include:

* Approximately 826K parameters
* Character-level tokenization
* Small training dataset
* 128-token context window
* Limited model capacity
* Limited general-world knowledge
* Limited semantic reasoning
* Overfitting during extended training
* Slower CPU inference
* No KV-cache implementation yet

The model should therefore not be directly compared with modern large-scale foundation models.

---

# 🚀 Future Improvements

## 1. Subword Tokenization

Replace character-level tokenization with BPE or another subword tokenizer.

This would allow the model to process common words and word fragments more efficiently.

---

## 2. Larger Context Window

Increase:

```text
128 tokens
```

to:

```text
256–512+ tokens
```

This would allow the model to retain more context during generation.

---

## 3. Larger Model

Increase:

* Transformer layers
* Embedding dimension
* Attention heads
* Total parameter count

This would give the model greater representational capacity.

---

## 4. Larger Dataset

Train on a larger and more diverse text corpus.

This would allow the model to learn broader vocabulary, grammar, and language patterns.

---

## 5. Improved Training

Future training improvements could include:

* Learning-rate warmup
* Mixed-precision training
* Early stopping
* Better checkpoint management
* More systematic evaluation
* Larger batch sizes where hardware allows

---

## 6. Faster Inference

Implement **KV caching** during autoregressive generation.

This would avoid repeatedly recomputing attention information for previously processed tokens and could significantly improve generation speed.

---

## 7. Better Sampling

Future versions could support:

* Top-P / nucleus sampling
* Repetition penalties
* More advanced decoding strategies

---

# 🎓 Key Concepts Learned

This project provided practical experience with the complete language-model pipeline:

* Tokenization
* Vocabulary construction
* Embeddings
* Positional embeddings
* Self-attention
* Query / Key / Value
* Multi-head attention
* Causal masking
* Transformer blocks
* Residual connections
* Layer normalization
* Feed-forward networks
* GELU
* Cross-entropy loss
* Backpropagation
* Gradients
* AdamW
* Weight decay
* Dropout
* Gradient clipping
* Learning-rate scheduling
* Perplexity
* Overfitting
* Validation
* Checkpointing
* Autoregressive generation
* Temperature sampling
* Top-K sampling
* CUDA GPU training
* Model inference
* Gradio
* Hugging Face Spaces
* ZeroGPU deployment

---

# 🎯 Project Goal

The goal of NovaGPT was **not to reproduce the capabilities of modern billion-parameter LLMs**.

The goal was to understand how a GPT-style language model works by implementing its core components from scratch and taking the project through the complete machine-learning lifecycle.

```text
                 RAW TEXT
                    │
                    ▼
              TOKENIZATION
                    │
                    ▼
               EMBEDDINGS
                    │
                    ▼
            TRANSFORMER BLOCKS
                    │
          ┌─────────┴─────────┐
          │                   │
     SELF-ATTENTION      FEED-FORWARD
          │                   │
          └─────────┬─────────┘
                    │
                    ▼
                LOGITS
                    │
                    ▼
             CROSS-ENTROPY
                    │
                    ▼
              BACKPROPAGATION
                    │
                    ▼
                 ADAMW
                    │
                    ▼
             TRAINED MODEL
                    │
                    ▼
              TEXT GENERATION
                    │
                    ▼
                INFERENCE
                    │
                    ▼
              GRADIO WEB APP
                    │
                    ▼
           HUGGING FACE SPACE
                    │
                    ▼
             PUBLIC AI DEMO
```

---

# 💡 Key Takeaway

NovaGPT demonstrates the complete process of building and deploying a small language model **without relying on a pretrained GPT model**.

The project combines:

> **Deep Learning + Transformers + PyTorch + CUDA + Model Training + Evaluation + Inference + Web Development + Cloud Deployment**

It represents an end-to-end implementation of a GPT-style language model, from raw text preprocessing all the way to a publicly accessible AI application.

---

## 👨‍💻 Author

**Your Name**

Built as a hands-on deep learning project to understand Transformer-based language models from first principles.
