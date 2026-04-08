# Therma: Discrete Thermodynamic Machine Emulation

Therma is a high-performance emulation framework designed to replace the deterministic Softmax sampling heads of Large Language Models (LLMs) with a Discrete Thermodynamic Machine (DTM).
Built on JAX, Therma re-interprets model weights as an energy landscape, utilizing stochastic relaxation and thermal noise to perform inference. This approach moves away from traditional "exact computation" toward a physical "state equilibrium" model, optimized for the next generation of analog-inspired AI hardware.

# The Core Tech
# 1. Weight-to-Energy Surgery
Therma performs a zero-shot projection of Transformer hidden states into a potential energy manifold. By treating weights as energy coefficients, we eliminate the need for expensive global normalization (Softmax) and replace it with local Gibbs sampling.

# 2. Dual-TSU Ping-Pong Architecture
To achieve high-throughput inference, Therma utilizes a twin Thermodynamic Sampling Unit (TSU) system:

Unit A (Sampling): Observes and reads the current token state.

Unit B (Relaxation): Simultaneously uses the latency window to thermally equilibrate the next token.

This architecture effectively hides the MCMC mixing time, enabling fluid token generation.

# 3. Stochastic Fidelity
Hardware Constraints: Emulates DAC precision limits and thermal noise floors.Beta 

($\beta$) Control: Dynamic inverse temperature scheduling to balance creativity and precision.

# Project Structure
├── core/               # JAX-based TSU & DTM engine
├── visualization/      # Interactive SVG/D3 energy manifold assets
├── Therma_Core.ipynb   # Full PoC: Weight Surgery & Inference
├── index.html          # Interactive Research Publication
└── LICENSE             # MIT

# Implementation
pip install jax[cuda12] transformers thrml

from therma import TSUCore

#Initialize Therma with a 512-node DTM
engine = TSUCore.load_emulation(backbone="Qwen2.5-0.5B")
output = engine.relax(input_text="The entropy of the system is")
