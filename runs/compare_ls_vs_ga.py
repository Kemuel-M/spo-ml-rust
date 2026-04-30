import os
import subprocess
import time
import csv
import re

# Configuration
RUST_BINARY = "./target/release/spo-ml-rust"
DATASETS_DIR = "datasets"
BEST_SOLUTIONS_FILE = "best_solutions/best_objectives.csv"
OUTPUT_CSV = os.path.join("outputs", "ls_vs_ga_comparison.csv")
TEMP_OUTPUT_DIR = "outputs/temp_comparison"

# Fixed Constructive Heuristic for all tests
FIXED_CONSTRUCTIVE = "aisle_adaptive"

# Strategies to compare
# Format: (Meta, Label)
META_STRATEGIES = [
    ("single", "LocalSearch"),
    ("ga", "MemeticGA")
]

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
    if ls_name == "swap":
        return "Hill Climbing", "Swap"
    elif ls_name == "insert":
        return "Hill Climbing", "Insertion"
    elif ls_name == "remove":
        return "Hill Climbing", "Removal"
    elif ls_name == "vnd":
        return "VND", "Mixed"
    elif ls_name == "tabu":
        return "Tabu Search", "Swap"
    elif ls_name == "lahc":
        return "LAHC", "Swap"
    elif ls_name == "none":
        return "None", "None"
    return "Unknown", "Unknown"

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
            ["python3", "checker.py", input_path, output_path],
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
    
    # Target instances for comparison (a subset to be faster)
    target_instances = [
        "instance_0001.txt", "instance_0002.txt", "instance_0003.txt",
        "instance_0006.txt", "instance_0009.txt", "instance_0012.txt"
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
            'instance', 'approach', 'ls_method', 'move_operator', 
            'found_value', 'best_known_value', 'gap_percent',
            'time_seconds', 'status'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        total_runs = len(instances) * len(META_STRATEGIES) * len(LOCAL_SEARCHES)
        current_run = 0
        
        print(f"Starting LS vs GA Comparison ({total_runs} runs)...")
        print(f"Using fixed constructive: {FIXED_CONSTRUCTIVE}")

        for dataset, filename, input_path in instances:
            best_known = best_objectives.get((dataset, filename), None)
            
            for meta, approach_label in META_STRATEGIES:
                for ls in LOCAL_SEARCHES:
                    current_run += 1
                    
                    # If meta is 'single' and ls is 'none', it's the baseline constructive
                    current_label = approach_label
                    if meta == "single" and ls == "none":
                        current_label = "Baseline (Constructive)"

                    strategy = f"{meta}+{FIXED_CONSTRUCTIVE}+{ls}"
                    temp_output_file = os.path.join(TEMP_OUTPUT_DIR, f"cmp_{meta}_{ls}_{dataset}_{filename}.txt")
                    method, operator = get_ls_details(ls)

                    print(f"[{current_run}/{total_runs}] {filename}: {meta}+{ls}...", end="", flush=True)
                    
                    start_time = time.time()
                    try:
                        # GA might take longer, using 120s timeout. Seed 42 for consistency.
                        proc = subprocess.run(
                            [RUST_BINARY, input_path, temp_output_file, strategy, "42"],
                            capture_output=True, text=True, timeout=150 
                        )
                        duration = time.time() - start_time
                        
                        if proc.returncode == 0:
                            is_feasible, score = validate_with_checker(input_path, temp_output_file)
                            if is_feasible:
                                gap = ""
                                if best_known and best_known > 0:
                                    gap = f"{((best_known - score) / best_known) * 100:.2f}%"
                                
                                print(f" Done. Score: {score:.4f}, Time: {duration:.2f}s")
                                writer.writerow({
                                    'instance': f"{dataset}/{filename}",
                                    'approach': current_label,
                                    'ls_method': method,
                                    'move_operator': operator,
                                    'found_value': f"{score:.4f}",
                                    'best_known_value': f"{best_known:.4f}" if best_known else "",
                                    'gap_percent': gap,
                                    'time_seconds': f"{duration:.4f}",
                                    'status': "OK"
                                })
                            else:
                                print(" Infeasible.")
                                writer.writerow({
                                    'instance': f"{dataset}/{filename}",
                                    'approach': current_label,
                                    'ls_method': method,
                                    'status': "INFEASIBLE"
                                })
                        else:
                            print(f" Failed ({proc.returncode})")
                            writer.writerow({
                                'instance': f"{dataset}/{filename}",
                                'approach': current_label,
                                'ls_method': method,
                                'status': f"FAILED({proc.returncode})"
                            })
                    except subprocess.TimeoutExpired:
                        print(" Timeout!")
                        writer.writerow({
                            'instance': f"{dataset}/{filename}", 
                            'approach': current_label,
                            'ls_method': method,
                            'status': "TIMEOUT"
                        })
                    except Exception as e:
                        print(f" Error: {e}")
                    
                    csvfile.flush()

    print(f"\nDone. Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
