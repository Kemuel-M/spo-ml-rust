import os
import subprocess
import time
import csv
import re
import sys

# Configuration
RUST_BINARY = "./target/release/spo-ml-rust"
DATASETS_DIR = "datasets"
BEST_SOLUTIONS_FILE = "best_solutions/best_objectives.csv"
OUTPUT_CSV = os.path.join("outputs", "benchmark_results_heuristics.csv")
TEMP_OUTPUT_DIR = "outputs/temp_benchmark"

# Metaheuristic is fixed to "single" for this benchmark (Constructive + LS only)
META = "single"

# All implemented constructive heuristics
CONSTRUCTIVES = [
    "static",
    "adaptive",
    "random",
    "aisle",
    "aisle_adaptive",
    "aisle_random"
]

# All implemented local search methods (Order-based and Aisle-based)
LOCAL_SEARCHES = [
    "none",
    "swap", "a_swap",
    "insert", "a_insert",
    "remove", "a_remove",
    "vnd", "a_vnd",
    "tabu", "a_tabu",
    "lahc", "a_lahc"
]

TIMEOUT_SECONDS = 120
SEED = 42

TARGET_INSTANCES = [
    "instance_0001.txt", "instance_0002.txt", "instance_0003.txt",
    "instance_0004.txt", "instance_0009.txt", "instance_0012.txt",
    "instance_0016.txt", "instance_0017.txt", "instance_0020.txt",
    "instance_0006.txt"
]

def get_ls_details(ls_name):
    """
    Returns (method, dimension, operator) based on the local search name.
    """
    dim = "Aisles" if ls_name.startswith("a_") else "Orders"
    base_name = ls_name.replace("a_", "")
    
    mapping = {
        "swap": ("Hill Climbing", "Swap"),
        "insert": ("Hill Climbing", "Insertion"),
        "remove": ("Hill Climbing", "Removal"),
        "vnd": ("VND", "Mixed"),
        "tabu": ("Tabu Search", "Swap/Mixed"),
        "lahc": ("LAHC", "Swap/Mixed"),
        "none": ("None", "None")
    }
    
    method, operator = mapping.get(base_name, ("Unknown", "Unknown"))
    if base_name == "none": dim = "None"
    
    return method, dim, operator

def compile_rust():
    print("Checking/Compiling Rust project...")
    result = subprocess.run(["cargo", "build", "--release"], capture_output=True, text=True)
    if result.returncode != 0:
        print("Compilation failed!")
        print(result.stderr)
        sys.exit(1)
    print("Compilation successful.")

def main():
    # Path adjustment if running from inside 'runs' directory
    global RUST_BINARY, DATASETS_DIR, BEST_SOLUTIONS_FILE, OUTPUT_CSV, TEMP_OUTPUT_DIR
    if os.path.basename(os.getcwd()) == "runs":
        os.chdir("..")

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    os.makedirs(TEMP_OUTPUT_DIR, exist_ok=True)

    compile_rust()

    # Find target instances in datasets/a
    instances = []
    ds_path = os.path.join(DATASETS_DIR, "a")
    if os.path.exists(ds_path):
        for file in sorted(os.listdir(ds_path)):
            if file in TARGET_INSTANCES:
                instances.append(("a", file, os.path.join(ds_path, file)))

    if not instances:
        print(f"No target instances found in {ds_path}")
        return

    print(f"Starting benchmark: {len(instances)} instances, {len(CONSTRUCTIVES)} constructives, {len(LOCAL_SEARCHES)} LS.")
    print(f"Total runs: {len(instances) * len(CONSTRUCTIVES) * len(LOCAL_SEARCHES)}")
    print(f"Timeout: {TIMEOUT_SECONDS}s | Seed: {SEED}")

    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        fieldnames = [
            'instance', 'heuristic', 'ls_method', 'ls_dimension', 
            'move_operator', 'found_value', 'time_seconds', 'status'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for ds_type, filename, input_path in instances:
            for constr in CONSTRUCTIVES:
                for ls in LOCAL_SEARCHES:
                    strategy = f"{META}+{constr}+{ls}"
                    temp_output_file = os.path.join(TEMP_OUTPUT_DIR, f"out_{ds_type}_{filename}_{constr}_{ls}.txt")
                    method, dimension, operator = get_ls_details(ls)

                    print(f"Running {ds_type}/{filename} | {strategy} ...", end=" ", flush=True)
                    
                    start_time = time.time()
                    try:
                        proc = subprocess.run(
                            [RUST_BINARY, input_path, temp_output_file, strategy, str(SEED)],
                            capture_output=True,
                            text=True,
                            timeout=TIMEOUT_SECONDS
                        )
                        duration = time.time() - start_time
                        
                        score = 0.0
                        status = "OK"
                        if proc.returncode == 0:
                            for line in proc.stdout.split('\n'):
                                if "Final Score:" in line:
                                    try:
                                        score = float(line.split(":")[1].strip())
                                    except: pass
                            
                            if "Solution NOT Feasible!" in proc.stdout:
                                status = "INFEASIBLE"
                            
                            print(f"Score: {score:.4f} ({duration:.2f}s)")
                        else:
                            status = f"ERROR_{proc.returncode}"
                            print(f"Failed (Exit code {proc.returncode})")

                        writer.writerow({
                            'instance': f"{ds_type}/{filename}",
                            'heuristic': constr,
                            'ls_method': method,
                            'ls_dimension': dimension,
                            'move_operator': operator,
                            'found_value': f"{score:.4f}",
                            'time_seconds': f"{duration:.4f}",
                            'status': status
                        })
                    except subprocess.TimeoutExpired:
                        print("Timeout!")
                        writer.writerow({
                            'instance': f"{ds_type}/{filename}",
                            'heuristic': constr,
                            'ls_method': method,
                            'ls_dimension': dimension,
                            'move_operator': operator,
                            'found_value': "0.0000",
                            'time_seconds': str(TIMEOUT_SECONDS),
                            'status': "TIMEOUT"
                        })
                    
                    csvfile.flush()

    print(f"\nBenchmark completed. Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
