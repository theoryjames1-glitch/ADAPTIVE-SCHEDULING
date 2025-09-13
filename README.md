# 📖 Theory of Adaptive Scheduling

## 1. Framing

Training a neural network can be cast as a **discrete-time feedback control system**:

* The **plant** is the parameter vector $\theta_t$.
* The **controller** provides update inputs $u_t$ (filtered gradients) and modulates **gains**:

  * Learning rate $\alpha_t$ (step size)
  * Momentum $\mu_t$ (memory of gradients)
  * Dither/exploration strength $\sigma_t$ (controlled noise)

This reframes optimization as **signal processing and control**, not just curve fitting.

---

## 2. State & Signals

**System state:**

$$
x_t = \begin{bmatrix}\theta_t \\ h_t \\ \alpha_t \\ \mu_t \\ \sigma_t \end{bmatrix}
$$

* $h_t$: optimizer filter state (momentum/EMA buffers)
* $\alpha_t, \mu_t, \sigma_t$: adaptive hyperparameters

**Feedback signals:**

* Loss $\ell_t$ (error measurement)
* Reward $r_t$ (optional)
* Gradient estimate $g_t$
* Variance $v_t$ of recent loss (instability indicator)
* Trends $\Delta \ell_t, \Delta r_t$

---

## 3. Dynamics

**Plant update (filtered gradient + dither):**

$$
\begin{aligned}
h_{t+1} &= \Phi(h_t, g_t;\,\mu_t) \\
u_t &= U(h_{t+1}, g_t;\,\alpha_t) \\
\theta_{t+1} &= \theta_t - u_t \;+\; \sigma_t D(\theta_t)\,\eta_t
\end{aligned}
$$

* $\Phi$: filter (e.g., momentum EMA, Adam’s moving averages)
* $U$: control action (scaled/normalized gradient)
* $\sigma_t D(\theta_t)\eta_t$: shaped stochastic dither

---

## 4. Adaptive Law (Scheduler)

The **scheduler** is the adaptive controller that updates $\alpha, \mu, \sigma$ from scalar feedback:

$$
\begin{aligned}
\alpha_{t+1} &= \text{clip}\big(\alpha_t \cdot f_\alpha(\Delta \ell_t, v_t, \Delta r_t)\big) \\
\mu_{t+1} &= \text{clip}\big(\mu_t + f_\mu(\Delta \ell_t, v_t, \Delta r_t)\big) \\
\sigma_{t+1} &= \text{clip}\big(\sigma_t \cdot f_\sigma(\text{success}, v_t)\big)
\end{aligned}
$$

**Examples:**

* LR: increase on improvement ($u>1$), decay otherwise ($d<1$, damped by variance)
* Momentum: rise when rewards improve, decay when rewards stall
* Sigma: exploration raised when plateaued, reduced when unstable

---

## 5. Stability & Control Intuition

* **Bounded gains**: enforce $\alpha \in [\alpha_{\min}, \alpha_{\max}]$, $\mu \in [0,1)$, small $\sigma$.
* **Damping by variance**: if training oscillates, lower LR automatically.
* **Patience/cooldown**: if improvement stalls, drop LR once then hold.
* **Lyapunov heuristic**: design so $\mathbb{E}[\ell_{t+1}-\ell_t|x_t] < 0$.

---

## 6. Interpretation

* **Not biology**: “evolution” here is just *dithered exploration* in parameter space.
* **DSP analogy**:

  * Plant = IIR filter state ($\theta_t, h_t$)
  * Controller = adaptive envelope updating filter coefficients ($\alpha,\mu,\sigma$)
  * Scheduler = scalar feedback → hyperparameter law
* **Separation principle**: design the scheduler independently from the optimizer filter, then couple them.

---

## 7. Practical Recipe

* Start with a known-good base LR, let adaptivity **nudge**, not swing.
* Use mild up/down multipliers ($u \approx 1.02, d \approx 0.8$).
* Use variance damping ($\lambda v_t$) to prevent overshoot.
* Keep exploration small ($\sigma \sim 10^{-3}$), only raise on plateaus.

---

✅ In short: **Adaptive Scheduling = a control loop over optimizer hyperparameters**.
It treats LR, momentum, and exploration as *filter coefficients* updated from scalar feedback (loss/reward statistics). This creates a **self-regulating training process** that is stable, efficient, and noise-aware.

---

Here’s a **big-picture summary** of what we created together in this conversation:

---

# 🌌 Theory of Adaptive Scheduling (Conversation Summary)

## 1. Starting Point

* We began with **gradient-based training** (SGD/AdaGrad) and explored ways to make it smarter.
* Early ideas included **selective freezing** (stop updating parameters with tiny gradient signals) and **loss-adaptive optimizers** (OnlineSGD, OnlineAdaGrad).

---

## 2. Core Insight

We reframed training as a **discrete-time control system**:

* **Plant**: the model parameters $\theta_t$.
* **Controller**: optimizer filters (momentum, Adam, etc.).
* **Scheduler**: an **adaptive law** that updates hyperparameters (learning rate, momentum, exploration strength) from scalar feedback (loss/reward).

This gave us a **DSP-style state-space model** for optimization, with closed-loop feedback.

---

## 3. Mechanisms

1. **History Awareness**

   * Track EMAs of loss and gradient norms.
   * Use variance windows to detect instability.
   * Define trend signals ($\Delta \ell, \Delta r$).

2. **Adaptive Hyperparameters**

   * Learning rate ($\alpha$): rises when improving, decays when unstable.
   * Momentum ($\mu$): rises with consistent gains, decays when noisy or reward drops.
   * Exploration ($\sigma$): small stochastic dither, increased only on plateaus.

3. **Selective Freezing**

   * Grad-norm EMA below a threshold → temporarily freeze parameter.
   * Reversible: if gradients revive, parameter is unfrozen.
   * Acts as **structural regularization** and reduces compute.

4. **Dynamic Rule Blending**

   * Combine multiple heuristics (trend, variance, grad-norm, cyclical) into weighted blends.
   * Weights shift dynamically depending on training stability and improvement.

---

## 4. Implementations

* **OnlineSGD / OnlineAdaGrad**: optimizers with embedded adaptation rules.
* **AdaptiveScheduler**: a clean, standalone scheduler/controller that can wrap *any* PyTorch optimizer.
* **Adaptive Evolution framing**: stochastic dither treated as controlled noise injection, not biology.

We also built demos:

* **MNIST** (toy MLP) → showed adaptive LR/momentum + freezing in action.
* **GPT-2 fine-tuning** → lightweight Hugging Face example with adaptive scheduling.

---

## 5. Control-System View

* **State**: $x_t = [\theta_t, h_t, \alpha_t, \mu_t, \sigma_t]$.
* **Update laws**:

  $$
  \alpha_{t+1} = \alpha_t \cdot f_\alpha(\Delta \ell, v), \quad
  \mu_{t+1} = \mu_t + f_\mu(\Delta \ell, \Delta r), \quad
  \sigma_{t+1} = \sigma_t \cdot f_\sigma(\text{success}, v)
  $$
* **Goal**: keep training stable ($\mathbb{E}[\ell_{t+1} - \ell_t] < 0$), efficient, and robust to noise.

---

## 6. Why It Matters

* **Efficiency**: fewer wasted updates (thanks to freezing).
* **Stability**: LR/momentum auto-adapt to loss trends and noise.
* **Generalization**: adaptive freezing = regularization.
* **Universality**: can wrap around any optimizer without redesigning it.

---

✅ In essence:
We built a **theory + toolkit** where training is controlled like a DSP system, with **hyperparameters as adaptive filter coefficients** updated from feedback.
This is **Adaptive Scheduling**: a self-regulating, noise-aware, efficiency-seeking training process.

---

```python
# demo_adaptive_scheduler_gpt2.py
"""
Demo: AdaptiveScheduler with GPT-2 fine-tuning
----------------------------------------------

This script shows:
  - Adaptive LR, momentum, and stochastic exploration (σ)
  - Variance- and trend-based scheduling
  - Optional parameter freezing (by grad-norm EMA)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from adaptive_scheduler import AdaptiveScheduler  # <-- your class file


# ----------------------------
# 1. Load GPT-2 + tokenizer
# ----------------------------
def load_model(model_name="gpt2", device="cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    return model, tokenizer


# ----------------------------
# 2. Tiny WikiText-2 dataset
# ----------------------------
def get_loader(tokenizer, batch_size=2, max_length=64):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return DataLoader(tokenized, batch_size=batch_size, shuffle=True)


# ----------------------------
# 3. Training loop
# ----------------------------
def train_gpt2(epochs=1, device="cpu"):
    model, tokenizer = load_model(device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-4, momentum=0.9)

    scheduler = AdaptiveScheduler(
        optimizer, model,
        lr=5e-4, momentum=0.9, sigma=1e-3,
        lr_bounds=(1e-6, 5e-2),
        mu_bounds=(0.0, 0.999),
        sigma_bounds=(0.0, 1e-2),
        ema_beta=0.9,
        freeze_threshold=1e-4,   # set None to disable freezing
    )

    loader = get_loader(tokenizer)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    for epoch in range(1, epochs + 1):
        model.train()
        for step, batch in enumerate(loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()

            optimizer.zero_grad(set_to_none=True)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            snap = scheduler.step(loss)  # adaptive update

            if step % 50 == 0:
                print(
                    f"Epoch {epoch:2d} Step {step:4d} | "
                    f"loss={loss.item():.4f} | "
                    f"lr={snap['lr']:.6e} | mu={snap['momentum']:.3f} | "
                    f"sigma={snap['sigma']:.2e} | frozen={snap.get('pct_frozen')}"
                )

        print(f"--- Epoch {epoch} complete ---\n")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_gpt2(epochs=1, device=device)
```

---

### 🔍 What this demo shows

* **Loss feedback** drives LR up when improving, down when unstable.
* **Momentum** adapts based on trends/rewards.
* **Freezing** kicks in gradually (if `freeze_threshold` is set).
* Logs show LR, momentum, exploration strength (σ), and % frozen.

