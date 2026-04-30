import os
import subprocess
import csv
import sys

# Configuration
INSTANCES = ["instance_0001.txt", "instance_0002.txt", "instance_0003.txt"]
CONSTRUCTIVES = ["static", "adaptive", "aisle_static", "aisle_adaptive"]
LOCAL_SEARCHES = ["none", "swap", "insert", "remove", "vnd", "tabu", "lahc"]

BINARY_PATH = "./target/release/spo-ml-rust"
INPUT_DIR = "datasets/a"
OUTPUT_DIR = "outputs/ls_comparison"

def run_solver(instance, constr, ls):
    input_path = os.path.join(INPUT_DIR, instance)
    output_path = os.path.join(OUTPUT_DIR, f"{instance}_{constr}_{ls}.txt")
    strategy = f"single+{constr}+{ls}"
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    cmd = [BINARY_PATH, input_path, output_path, strategy]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if "Final Score:" in line:
                    try:
                        return float(line.split(":")[1].strip())
                    except:
                        pass
        return 0.0
    except Exception as e:
        return 0.0

def main():
    print(f"{'Instance':<15} {'Constr':<15} {'None':<10} {'Best LS':<10} {'Improvement':<12}")
    print("-" * 80)
    
    results = []
    
    for inst in INSTANCES:
        for constr in CONSTRUCTIVES:
            none_score = run_solver(inst, constr, "none")
            
            best_ls_score = 0.0
            best_ls_name = "none"
            
            ls_scores = {}
            for ls in LOCAL_SEARCHES:
                if ls == "none": continue
                score = run_solver(inst, constr, ls)
                ls_scores[ls] = score
                if score > best_ls_score:
                    best_ls_score = score
                    best_ls_name = ls
            
            improvement = 0.0
            if none_score > 0:
                improvement = ((best_ls_score - none_score) / none_score) * 100
            elif best_ls_score > 0:
                improvement = 100.0
                
            print(f"{inst:<15} {constr:<15} {none_score:<10.4f} {best_ls_score:<10.4f} {improvement:>10.2f}% ({best_ls_name})")
            
            results.append({
                'instance': inst,
                'constructive': constr,
                'none': none_score,
                'best_ls': best_ls_score,
                'best_ls_name': best_ls_name,
                'improvement': improvement,
                'all_ls': ls_scores
            })

    # Summary analysis
    total_improvement = sum(r['improvement'] for r in results if r['none'] > 0)
    count = sum(1 for r in results if r['none'] > 0)
    if count > 0:
        print("-" * 80)
        print(f"Average Improvement on valid base solutions: {total_improvement/count:.2f}%")

if __name__ == "__main__":
    if not os.path.exists(BINARY_PATH):
        print("Binary not found. Please run cargo build --release first.")
        sys.exit(1)
    main()
