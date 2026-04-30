import os
import subprocess
import time
import csv
import re

RUST_BINARY = "./target/release/spo-ml-rust"
DATASETS_DIR = "datasets"
OUTPUT_CSV = os.path.join("outputs", "ga_consistency_check.csv")
TEMP_OUTPUT_DIR = "outputs/temp_ga_check"

BEST_CONSTRUCTIVE = "aisle_random"
META = "ga"
STRATEGIES = ["none", "vnd"]

def validate_with_checker(input_path, output_path):
    try:
        proc = subprocess.run(
            ["python3", "checker.py", input_path, output_path],
            capture_output=True, text=True, timeout=10
        )
        if proc.returncode != 0: return False, None
        is_feasible = "Is solution feasible: True" in proc.stdout
        match = re.search(r"Objective function value: (\d+\.\d+)", proc.stdout)
        score = float(match.group(1)) if match else None
        return is_feasible, score
    except Exception:
        return False, None

def main():
    os.makedirs(TEMP_OUTPUT_DIR, exist_ok=True)
    target_instances = [
        "instance_0001.txt", "instance_0002.txt", "instance_0003.txt", 
        "instance_0004.txt", "instance_0009.txt", "instance_0012.txt", 
        "instance_0016.txt", "instance_0017.txt", "instance_0020.txt"
    ]
    
    instances = []
    for root, dirs, files in os.walk(os.path.join(DATASETS_DIR, "a")):
        for file in files:
            if file in target_instances:
                instances.append((file, os.path.join(root, file)))
    instances.sort()

    results = {}

    print(f"Starting Consistency Check: GA+None vs GA+VND...")

    for filename, input_path in instances:
        results[filename] = {}
        for ls in STRATEGIES:
            strategy = f"{META}+{BEST_CONSTRUCTIVE}+{ls}"
            temp_out = os.path.join(TEMP_OUTPUT_DIR, f"check_{filename}_{ls}.txt")
            
            print(f"Running {filename} [{strategy}]...", end="", flush=True)
            start_time = time.time()
            try:
                proc = subprocess.run(
                    [RUST_BINARY, input_path, temp_out, strategy],
                    capture_output=True, text=True, timeout=120
                )
                duration = time.time() - start_time
                feasible, score = validate_with_checker(input_path, temp_out)
                
                if feasible:
                    results[filename][ls] = score
                    print(f" OK: {score:.4f} ({duration:.1f}s)")
                else:
                    results[filename][ls] = 0.0
                    print(f" INFEASIBLE")
            except Exception as e:
                print(f" ERROR: {e}")
                results[filename][ls] = 0.0

    print("\n--- FINAL COMPARISON ---")
    print(f"{'Instance':<20} | {'GA+None':<10} | {'GA+VND':<10} | {'Improvement'}")
    print("-" * 60)
    
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["instance", "ga_none", "ga_vnd", "diff"])
        
        for inst in target_instances:
            none_score = results.get(inst, {}).get("none", 0.0)
            vnd_score = results.get(inst, {}).get("vnd", 0.0)
            diff = vnd_score - none_score
            gain = "UP" if diff > 0.0001 else ("EQUAL" if abs(diff) < 0.0001 else "DOWN")
            
            print(f"{inst:<20} | {none_score:<10.4f} | {vnd_score:<10.4f} | {gain}")
            writer.writerow([inst, none_score, vnd_score, diff])

if __name__ == "__main__":
    main()
