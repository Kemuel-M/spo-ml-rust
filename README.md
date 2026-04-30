# SPO-ML-Rust: Wave Order Picking Solver

This project is a Rust-based command-line application designed to solve the "Wave Order Picking" combinatorial optimization problem. The goal is to select a subset of customer orders (a "wave") to maximize picking productivity.

The objective function is defined as:
**Productivity = Total Items Picked / Total Aisles Visited**

The solver implements a modular architecture allowing users to combine different Constructive Heuristics, Local Search methods, and Metaheuristics via the command line.

## Project Structure

```
.
├── Cargo.toml            # Rust project manifest
├── checker.py            # Python script to validate a solution and calculate its score
├── run_challenge.py      # Main Python script to compile, run, and evaluate the solver
├── src/
│   ├── main.rs           # Main application entry point, CLI argument parsing
│   ├── solver.rs         # Core solver logic, handles strategy building
│   ├── heuristics/       # Constructive heuristics and the modular Timer wrapper
│   ├── local_searchs/    # Hill Climbing, VND, Tabu Search, LAHC implementations
│   └── metaheuristics/   # GRASP, ILS, Simulated Annealing
├── datasets/             # Input instance files
```

## Performance Measurement (Timing)

The solver includes a modular timing system using a **Decorator Pattern**. This allows measuring the performance of any phase of the algorithm without modifying its core logic.

When running the solver, you will see output like:
- `[Timer] Construction (...) took X.XXXXs`: Time spent building the initial solution.
- `[Timer] Local Search (...) took X.XXXXs`: Time spent refining the solution.
- `[Timer] >>> Total Execution Time: X.XXXXs <<<`: Wall-clock time for the entire strategy.

This is useful for comparing the efficiency of different metaheuristics and local search algorithms.

## How to Run and Evaluate

### 1. Using the Automated Script (Recommended)

The `run_challenge.py` script automates compilation, execution, and validation.

**Usage:**
```bash
python3 run_challenge.py <project_root> <input_path> [strategy_config]
```

**Examples:**

1.  **Run a specific strategy on one instance:**
    ```bash
    python3 run_challenge.py . datasets/a/instance_0001.txt grasp+random+vnd
    ```

2.  **Run a set of showcase strategies on a full dataset:**
    ```bash
    python3 run_challenge.py . datasets/a all
    ```
    *This will run variations like `single+static+none`, `grasp+random+vnd`, etc.*

### 2. Manual Execution (Rust Binary)

**Compilation:**
```bash
cargo build --release
```

**Usage:**
```bash
./target/release/spo-ml-rust <input_file> <output_file> <strategy_config>
```

**Example:**
```bash
./target/release/spo-ml-rust datasets/a/instance_0001.txt out.txt ils+aisle_random+lahc
```

## Strategy Configuration Guide

The solver now accepts a configuration string in the format:
`META+CONSTRUCTIVE+LOCALSEARCH`

You can mix and match components to create custom solvers.

### 1. META (The Orchestrator)
Controls the high-level flow of the optimization.
- `single`: Runs Construction -> Local Search (once).
- `grasp`: Greedy Randomized Adaptive Search Procedure (Iterations=50).
- `ils`: Iterated Local Search (Iterations=20).
- `sa`: Simulated Annealing.

### 2. CONSTRUCTIVE (The Initial Solution)
Generates the starting solution.
- `static`: Deterministic Greedy (Orders sorted by density).
- `adaptive`: Adaptive Greedy (Re-evaluates costs at each step).
- `random`: Randomized Greedy (Based on `alpha` parameter).
- `aisle`: Aisle-Centric Static (Prioritizes opening dense aisles).
- `aisle_adaptive`: Aisle-Centric Adaptive.
- `aisle_random`: Aisle-Centric Randomized.

### 3. LOCALSEARCH (The Refinement)
Improves the solution.
- `none`: No local search.
- `vnd`: Variable Neighborhood Descent (Insertion -> Swap -> Removal).
- `tabu`: Tabu Search.
- `lahc`: Late Acceptance Hill Climbing.
- `swap`: Simple Hill Climbing (Swap neighborhood).
- `insert`: Simple Hill Climbing (Insertion neighborhood).
- `remove`: Simple Hill Climbing (Removal neighborhood).

### Common Combinations (Recipes)

| Strategy String | Description |
| :--- | :--- |
| `single+static+none` | Baseline: Pure Static Greedy |
| `single+adaptive+vnd` | High Quality Deterministic: Adaptive construction + VND refinement |
| `grasp+random+vnd` | Standard Metaheuristic: GRASP with VND |
| `ils+aisle_random+lahc` | Experimental: ILS using Aisle logic and Late Acceptance |

## Validating Results

The `checker.py` script verifies solution feasibility (stock limits, wave size) and computes the final score.

```bash
python3 checker.py <input_file> <output_file>
```
