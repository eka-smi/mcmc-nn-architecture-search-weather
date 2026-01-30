# MCMC Neural Architecture Search for Exam-Day Weather Forecasting

This repository implements **Markov Chain Monte Carlo (MCMC)** optimization (Metropolis–Hastings) for selecting an optimal **fully-connected neural network (MLP)** architecture via a random walk in the space of configurations, and applies it to **weather forecasting** using historical meteorological station data.

---

## 1) Detailed Task Description (Task 4: MCMC-based Architecture Optimization)

Selecting an optimal neural network architecture by brute force is infeasible because the number of configurations grows exponentially with depth and layer width.  
Instead of enumerating all architectures, this project performs **stochastic search** over configurations using **MCMC (Metropolis–Hastings)**.

### Architecture representation
An architecture is represented as an ordered list of integers:
- `k ≤ K` — number of hidden layers  
- `[n1, n2, ..., nk]` — number of neurons per hidden layer (bounded by `M`)  
Optionally, the architecture may also include:
- **activation choices** per layer
- **depth changes** (adding/removing a layer)

### Elementary step (proposal move)
At each MCMC iteration the algorithm proposes a small local modification of the current architecture, for example:
- **Width move:** pick a layer and increase/decrease its neuron count
- **Activation move:** change activation function for one layer
- **Depth move:** add a new hidden layer or remove an existing one (within limits)

These moves define a random walk on the discrete architecture space.

### Objective / “energy” function
Each candidate architecture is trained and evaluated. Its quality is mapped to an energy value `E(arch)`, such as:
- validation loss (lower is better)
- or negative validation score (higher is better → lower energy)

### Metropolis–Hastings acceptance rule
A proposed architecture is accepted with probability:
\[
\alpha = \min\left(1, \exp\left(-\frac{E_{new} - E_{old}}{T}\right)\right)
\]
where `T` is an optional temperature-like parameter.

Importantly, **worse architectures are not always rejected**.  
If the new proposal has higher energy (e.g., larger validation loss), it can still be accepted with a non-zero probability. This controlled acceptance of worse moves helps the chain **avoid getting stuck in local minima** and encourages exploration of the architecture space.


### Training budget comparison
The implementation is designed to compare two realistic evaluation regimes:
- **fixed number of epochs** for every candidate architecture
- **fixed training time** per candidate architecture

---

## 2) Application (Task: Weather Forecast for a specific day)

### Goal
Using weather observations from the **previous 7 days (excluding the X-day)**, the model predicts at **15:00 on the X-day**:
- **temperature**
- **wind**
- **humidity**

### Data source
The input data comes from an **archival meteorological station export** (XLS format from https://rp5.ru/Архив_погоды_в_Москве_(ВДНХ)).  
The dataset contains multi-year observations with meteorological variables such as:
- temperature, dew point, humidity
- pressure (station/sea-level representations)
- wind direction and speed, gust-related fields
- cloud cover, cloud types/heights, visibility
- precipitation amount/duration and weather condition codes

The raw file is included in the repository and used for feature engineering and reproducible experiments.

### Uncertainty estimation
To estimate uncertainty, the project supports ensemble-style inference such as:
- **multiple stochastic forward passes** (e.g., dropout-based sampling)
- aggregating predictions across multiple model runs  
to produce **prediction intervals**.

---

## 3) Technologies Used

The implementation is written in Python and uses:

- **Python (standard library)**: `math`, `random`, `time`, `datetime`, `timedelta`, `os`, `re`, `dataclasses`, `typing`
- **NumPy**: numerical computations and array operations
- **Pandas**: reading and preprocessing weather data tables, time-based feature engineering
- **PyTorch**: neural network definition (`torch.nn`), training loops, and dataset utilities
  - `TensorDataset`, `DataLoader` for batching and training

---

## 4) Program Workflow (How the system works)

**b. MCMC architecture optimization (Metropolis–Hastings)**

    - Starts from an initial valid architecture under given constraints (depth, neurons, activations).
    - Repeats: proposes a neighboring architecture → trains it under a fixed budget → evaluates on validation.
    - Accepts better candidates and sometimes worse ones (to escape local optima), while tracking the best architecture seen.
    
**c. Applying the algorithm to a specific task**

    - Loads and preprocesses the dataset, then defines the target metric.
    - Runs the MCMC loop on this task and outputs the best architecture + key metrics.

**d. Two evaluation budgets: fixed epochs vs fixed time**

    - Fixed epochs: every candidate trains for the same number of epochs (consistent training steps).
    - Fixed time: every candidate trains for the same wall‑clock time (fair compute budget).

      The program can compare the best results achieved in both modes.

**e. Optional move: change a layer activation**

    - Randomly selects a layer and swaps its activation from an allowed set (e.g., ReLU ↔ Tanh).
    - This lets the search improve training dynamics without changing layer sizes.

**f. Optional move: change network depth**

    - Proposes adding or removing a hidden layer (within min/max depth limits).
    - This enables exploring “shallow‑wide” vs “deep‑narrow” architectures during the same search.

