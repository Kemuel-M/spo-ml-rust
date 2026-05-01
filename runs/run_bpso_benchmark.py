import os
import subprocess
import time
import csv
import re

# Configuration
RUST_BINARY = "./target/release/spo-ml-rust"
DATASETS_DIR = "datasets"
BEST_SOLUTIONS_FILE = "best_solutions/best_objectives.csv"
OUTPUT_CSV = os.path.join("outputs", "bpso_benchmark.csv")
TEMP_OUTPUT_DIR = "outputs/temp_bpso"

# Fixed Best Constructive
BEST_CONSTRUCTIVE = "aisle_random"

# Metaheuristic base
META_BASE = "bpso"

# Dimensions to test
DIMENSIONS = ["orders", "aisles"]

# Local Searches to test as the Memetic step
LOCAL_SEARCHES = [
    "none",
    "vnd",
    "tabu",
    "lahc",
    "swap",
    "insert",
    "remove"
]

def get_ls_details(ls_name):
    clean_ls = ls_name.replace("a_", "")
    if clean_ls == "swap":
        return "Hill Climbing", "Swap", "Best Improvement"
    elif clean_ls == "insert":
        return "Hill Climbing", "Insertion", "Best Improvement"
    elif clean_ls == "remove":
        return "Hill Climbing", "Removal", "Best Improvement"
    elif clean_ls == "vnd":
        return "VND", "Mixed (Ins->Swap->Rem)", "Variable Neighborhood"
    elif clean_ls == "tabu":
        return "Tabu Search", "Swap", "Tabu Logic"
    elif clean_ls == "lahc":
        return "LAHC", "Swap", "Late Acceptance"
    elif clean_ls == "none":
        return "None", "None", "None"
    return "Unknown", "Unknown", "Unknown"

def load_best_objectives():
    best_objs = {}
    if not os.path.exists(BEST_SOLUTIONS_FILE):
        return best_objs
    with open(BEST_SOLUTIONS_FILE, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 3:
                dataset, instance, value = row[0], row[1], row[2]
                best_objs[(dataset, instance)] = float(value)
    return best_objs

def validate_with_checker(input_path, output_path):
    try:
        proc = subprocess.run(
            ["python3", "runs/checker.py", input_path, output_path],
            capture_output=True,
            text=True,
            timeout=10
        )
        if proc.returncode != 0: return False, None
        is_feasible = "Is solution feasible: True" in proc.stdout
        if not is_feasible: return False, None
        match = re.search(r"Objective function value: (\d+\.\d+)", proc.stdout)
        score = float(match.group(1)) if match else None
        return True, score
    except Exception:
        return False, None

def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    os.makedirs(TEMP_OUTPUT_DIR, exist_ok=True)
    
    best_objectives = load_best_objectives()
    
    # Target instances for faster benchmarking
    target_instances = [
        "instance_0001.txt", "instance_0002.txt", "instance_0003.txt",
        "instance_0004.txt", "instance_0009.txt", "instance_0012.txt",
        "instance_0016.txt", "instance_0017.txt", "instance_0020.txt",
        "instance_0006.txt"
    ]
    
    instances = []
    for root, dirs, files in os.walk(os.path.join(DATASETS_DIR, "a")):
        dataset = "a"
        for file in files:
            if file in target_instances:
                path = os.path.join(root, file)
                instances.append((dataset, file, path))
    instances.sort()

    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        fieldnames = [
            'instance', 'dimension', 'heuristic', 'ls_method', 'move_operator', 
            'acceptance_policy', 'found_value', 'best_known_value', 
            'time_seconds', 'total_iterations', 'status'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        total_runs = len(instances) * len(DIMENSIONS) * len(LOCAL_SEARCHES)
        current_run = 0
        
        print(f"Starting BPSO Benchmark with Dimensions ({total_runs} runs)...")

        for dataset, filename, input_path in instances:
            best_known = best_objectives.get((dataset, filename), "")
            
            for dim in DIMENSIONS:
                meta = "bpso" if dim == "orders" else "a_bpso"
                
                for ls_base in LOCAL_SEARCHES:
                    current_run += 1
                    
                    # Use dimension-specific local search if not "none"
                    ls = ls_base if (dim == "orders" or ls_base == "none") else f"a_{ls_base}"
                    
                    strategy = f"{meta}+{BEST_CONSTRUCTIVE}+{ls}"
                    temp_output_file = os.path.join(TEMP_OUTPUT_DIR, f"out_bpso_{dim}_{dataset}_{filename}_{ls}.txt")
                    method, operator, policy = get_ls_details(ls)

                    print(f"[{current_run}/{total_runs}] Running {filename} ({dim}) with {meta} + {ls}...", end="", flush=True)
                    
                    start_time = time.time()
                    try:
                        proc = subprocess.run(
                            [RUST_BINARY, input_path, temp_output_file, strategy, "42"],
                            capture_output=True, text=True, timeout=120 
                        )
                        duration = time.time() - start_time
                        
                        # Extract total iterations
                        iter_match = re.search(r"\[BPSO\] Finished in (\d+) iterations", proc.stdout)
                        total_iters = iter_match.group(1) if iter_match else ""

                        if proc.returncode == 0:
                            is_feasible, score = validate_with_checker(input_path, temp_output_file)
                            if is_feasible:
                                print(f" Done. Score: {score:.4f}, Iters: {total_iters}, Time: {duration:.2f}s")
                                writer.writerow({
                                    'instance': f"{dataset}/{filename}",
                                    'dimension': dim,
                                    'heuristic': meta.upper(),
                                    'ls_method': method,
                                    'move_operator': operator,
                                    'acceptance_policy': policy,
                                    'found_value': f"{score:.4f}",
                                    'best_known_value': best_known,
                                    'time_seconds': f"{duration:.4f}",
                                    'total_iterations': total_iters,
                                    'status': "OK"
                                })
                            else:
                                print(" Infeasible.")
                                writer.writerow({
                                    'instance': f"{dataset}/{filename}",
                                    'dimension': dim,
                                    'heuristic': meta.upper(),
                                    'ls_method': method,
                                    'found_value': "",
                                    'total_iterations': total_iters,
                                    'status': "INFEASIBLE"
                                })
                        else:
                            print(f" Failed ({proc.returncode})")
                    except subprocess.TimeoutExpired:
                        print(" Timeout!")
                        writer.writerow({'instance': filename, 'dimension': dim, 'status': "TIMEOUT"})
                    except Exception as e:
                        print(f" Error: {e}")
                    
                    csvfile.flush()

    print(f"\nDone. Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
