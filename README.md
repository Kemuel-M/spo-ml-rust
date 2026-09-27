# SPO-ML-Rust: Wave Order Picking Solver

A high-performance Rust solver for the **Wave Order Picking** combinatorial optimization problem (originally proposed for the Mercado Libre First Optimization Challenge at SBPO).

The goal is to select a subset of customer orders (a "wave") to maximize picking productivity while respecting warehouse stock capacities and wave size limits:

$$\text{Productivity} = \frac{\text{Total Items Picked}}{\text{Total Aisles Visited}}$$

The solver features a modular architecture enabling flexible composition of **Constructive Heuristics**, **Local Search neighborhoods**, and **Metaheuristics** via command-line configuration strings.

---

## Project Structure

```text
.
├── Cargo.toml            # Rust package manifest (spo-ml-rust, bench binaries)
├── runs/                 # Automation, validation, and benchmarking scripts
│   ├── checker.py        # Python script to validate solution feasibility & score
│   ├── run_challenge.py  # Automation script to compile, run, and evaluate instances
│   ├── benchmark_constructives.py # Benchmark suite for constructive heuristics
│   ├── benchmark_local_search.py  # Benchmark suite for local search algorithms
│   ├── benchmark_h_grasp.py       # Benchmark suite for Reactive GRASP + HVND
│   └── benchmark_h_bpso.py        # Benchmark suite for Hybrid BPSO + HVND
├── src/
│   ├── main.rs           # Application entry point and CLI argument parser
│   ├── bench.rs          # Standalone benchmark binary for performance profiling
│   ├── lib.rs            # Library entry point exporting modules
│   ├── io.rs             # High-throughput instance reader and solution writer
│   ├── solution.rs       # ProblemData and ChallengeSolution core structures
│   ├── solver/           # Solver orchestration and strategy builder
│   │   ├── mod.rs        # ChallengeSolver, builds execution graph
│   │   ├── config.rs     # Parser for strategy strings, hyperparameters & defaults
│   │   └── simple_solver.rs # Single-shot pipeline (Constructive -> Local Search)
│   ├── evaluator/        # Core incremental evaluator & SIMD/AVX2 stock manager
│   │   ├── types.rs      # Evaluator trait and Move definitions
│   │   ├── core_stock.rs # Vectorized global stock balance tracking
│   │   └── stock_balance.rs # Incremental evaluator with rollback & aisle pruning
│   ├── heuristics/       # Constructive heuristics & modular Timer wrapper
│   ├── local_searchs/    # Local search algorithms (VND, HVND, Tabu, LAHC, HC)
│   └── metaheuristics/   # Metaheuristics (GRASP, ILS, SA, GA, BPSO)
├── datasets/             # Input instance sets (a, b, x)
└── outputs/              # Generated solution files and benchmark reports (CSV)
```

---

## Performance Measurement (Timing)

The solver incorporates a modular timing system using the **Decorator Pattern** (`TimedConstructive`, `TimedLocalSearch`, `TimedSolver`). This measures execution phases without altering algorithm internals.

When running single-pass strategies, output includes:
- `  [Timer] Construction (...) took X.XXXXs`: Time spent building the initial solution.
- `  [Timer] Local Search (...) took X.XXXXs (Improved: true/false)`: Time spent refining the solution.
- `  [Timer] Total Solver (...) took X.XXXXs`: Total wall-clock time for the full strategy.

*(Note: In iterative metaheuristics such as GRASP, BPSO, GA, and ILS, granular per-iteration timers are suppressed to keep logs clean; only total strategy time is displayed.)*

---

## How to Run

### 1. Manual Execution (Rust Binary)

**Build Release:**
```bash
cargo build --release
```

**CLI Syntax:**
```bash
./target/release/spo-ml-rust <INPUT> <OUTPUT> [STRATEGY_CONFIG] [SEED]
```

**Arguments:**
- `<INPUT>`: Path to instance file (e.g., `datasets/a/instance_0001.txt`).
- `<OUTPUT>`: Destination path for solution file (e.g., `outputs/out.txt`).
- `[STRATEGY_CONFIG]` *(optional, default: `single+order_static+none`)*: Strategy string formatted as `META+CONSTRUCTIVE+LOCALSEARCH`.
- `[SEED]` *(optional, default: `42`)*: PRNG seed as a `u64`. Pass `0` to seed from system time.

**Examples:**
```bash
# Deterministic baseline with default seed (42)
./target/release/spo-ml-rust datasets/a/instance_0001.txt out.txt single+order_static+none

# High-quality deterministic strategy
./target/release/spo-ml-rust datasets/a/instance_0001.txt out.txt single+order_adaptive+hvnd

# Reactive GRASP with Hybrid VND and random seed
./target/release/spo-ml-rust datasets/a/instance_0001.txt out.txt grasp+order_random+hvnd 0

# Hybrid BPSO with SuperHybrid constructive and timeout override (600s)
./target/release/spo-ml-rust datasets/x/instance_0010.txt out.txt h_bpso:1000:100:600+sh:0.05+hvnd 0
```

---

### 2. Using Python Scripts (`runs/`)

#### Solution Feasibility Checker
Validates solution constraints (stock availability, wave size bounds) and computes the exact score:
```bash
python3 runs/checker.py <input_file> <solution_file>
```

#### Challenge Runner
Automates compilation, execution, and validation over instances or folders:
```bash
python3 runs/run_challenge.py . datasets/a single+order_adaptive+hvnd
```

#### Specialized Benchmark Suites
The `runs/` folder contains ready-to-run benchmark scripts that generate CSV reports in `outputs/`:
- `runs/benchmark_constructives.py`: Compares all constructive heuristics over datasets.
- `runs/benchmark_local_search.py`: Evaluates local search strategies from initial baselines.
- `runs/benchmark_h_grasp.py`: Benchmarks Reactive GRASP configurations.
- `runs/benchmark_h_bpso.py`: Benchmarks Hybrid BPSO configurations.

---

## Strategy Configuration Guide

Strategies follow the format:
```text
META+CONSTRUCTIVE+LOCALSEARCH
```

Component strings can use either long names or short aliases.

### 1. META (The Orchestrator)

Controls the high-level optimization lifecycle. Supports optional parameter overrides in the format `META[:iter[:pop[:timeout_secs]]]`.

| Name / Alias | Description | Default Parameters |
| :--- | :--- | :--- |
| `single` | Executes Construction followed by Local Search once. | N/A |
| `grasp` | Reactive GRASP (Greedy Randomized Adaptive Search Procedure). | 100 iterations, 150s timeout |
| `ils` | Iterated Local Search with perturbation and stagnation restart. | 100 iterations, 150s timeout |
| `sa` | Simulated Annealing with adaptive cooling, reheating, and kick. | $T_0=100$, cooling=0.999 |
| `ga` | Genetic Algorithm / Memetic Algorithm with elitism and repair. | 100 gen, pop=50 |
| `bpso` | Binary Particle Swarm Optimization in Order space. | 1000 iter, pop=100, 600s |
| `h_bpso` | Hybrid BPSO combining RK continuous decoding and order spaces. | 1000 iter, pop=100, 600s |
| `a_*` | Aisle-dimension variants (`a_grasp`, `a_ils`, `a_sa`, `a_ga`, `a_bpso`). | Same as base |

*Tip: Add a 3rd colon parameter to override timeout, e.g., `grasp:200::300` or `h_bpso:1000:100:600`.*

---

### 2. CONSTRUCTIVE (The Initial Solution)

Generates an initial feasible solution.

| Identifier | Short Alias | Description |
| :--- | :--- | :--- |
| `order_static` | `os` | Deterministic Greedy: Orders sorted by item density. |
| `order_adaptive` | `oa` | Adaptive Greedy: Recalculates marginal aisle impact at each step. |
| `order_random` | `or` | Randomized Greedy: RCL based on dynamic $\alpha$ candidate list. |
| `aisle_static` | `as` | Aisle-Centric Static: Greedily opens dense aisles first. |
| `aisle_adaptive` | `aa` | Aisle-Centric Adaptive: Iteratively selects best marginal aisle. |
| `aisle_random` | `ar` | Aisle-Centric Randomized: Semi-greedy aisle selection with RCL. |
| `hybrid_random` | `hr` | Hybrid Random: Interleaves order and aisle greedy steps. |
| `super_hybrid` | `sh` *(or `sh:p_zero`)* | Super Hybrid: Supports $\alpha=0$ intensification bias (e.g., `sh:0.05`). |

---

### 3. LOCALSEARCH (The Refinement)

Refines solutions using neighborhood exploration.

| Identifier | Description |
| :--- | :--- |
| `none` | No local search applied. |
| `vnd` | Variable Neighborhood Descent in Order space (`Insertion` $\to$ `Swap` $\to$ `Removal`). |
| `a_vnd` | Variable Neighborhood Descent in Aisle space. |
| `hvnd` | **Hybrid VND**: Combines Order (`Insertion`, `Swap`) and Aisle (`Removal`, `Swap`) moves. |
| `tabu` *(or `tabu:iter`)* | Tabu Search with tenure=20 in Order space (e.g., `tabu:500`). |
| `a_tabu` | Tabu Search in Aisle space. |
| `lahc` *(or `lahc:iter`)* | Late Acceptance Hill Climbing ($L=50$) in Order space. |
| `a_lahc` | Late Acceptance Hill Climbing in Aisle space. |
| `hlahc` *(or `hlahc:iter`)*| **Hybrid LAHC**: Late Acceptance over combined order/aisle moves. |
| `swap` / `a_swap` | Hill Climbing using Swap neighborhood (Best Improvement). |
| `hhc` | **Hybrid Hill Climbing**: Multi-neighborhood hill climbing. |

---

### Recommended Recipes

| Strategy String | Characteristics |
| :--- | :--- |
| `single+order_static+none` | **Ultra-Fast Baseline**: Pure deterministic greedy (~0.01s). |
| `single+order_adaptive+none` | **High-Quality Deterministic**: Marginal re-evaluation without local search. |
| `single+order_adaptive+hvnd` | **State-of-the-Art Single Pass**: Fast and strong deterministic combination. |
| `grasp+order_random+hvnd` | **Standard Metaheuristic**: Reactive GRASP with Hybrid VND refinement. |
| `h_bpso+sh:0.05+hvnd` | **Top Benchmark Performer**: Hybrid BPSO using SuperHybrid and Hybrid VND. |
