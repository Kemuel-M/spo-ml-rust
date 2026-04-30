import subprocess
import os
import time
import sys

# Configuration
INSTANCES = [
    "instance_0001.txt", "instance_0002.txt", "instance_0003.txt", 
    "instance_0004.txt", "instance_0009.txt", "instance_0012.txt", 
    "instance_0016.txt", "instance_0017.txt", "instance_0020.txt"
]

METAS = [
    "grasp",
    "ils",
    "sa"
]

CONSTRUCTIVES = [
    "static",
    "adaptive",
    "random",
    "aisle_static",
    "aisle_adaptive",
    "aisle_random"
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

BINARY_PATH = "./target/release/spo-ml-rust"
INPUT_DIR = "datasets/a"
OUTPUT_DIR = "outputs/test_meta_run"

# Colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BLUE = "\033[94m"
CYAN = "\033[96m"

def run_test():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Calculate total runs
    # SA runs 1 LS (none) * 6 Constr = 6 per instance
    # GRASP/ILS run 7 LS * 6 Constr = 42 per instance
    # Total per instance = 42 + 42 + 6 = 90
    # Total = 90 * 9 = 810
    
    print(f"Starting Metaheuristic tests.")
    print("-" * 80)

    failed_tests = []
    
    for instance in INSTANCES:
        print(f"\n{BLUE}Instance: {instance}{RESET}")
        
        for meta in METAS:
            for constr in CONSTRUCTIVES:
                # Optimization: SA ignores LS, so only run once with 'none'
                current_ls_list = ["none"] if meta == "sa" else LOCAL_SEARCHES
                
                for ls in current_ls_list:
                    input_path = os.path.join(INPUT_DIR, instance)
                    output_file = f"{instance.replace('.txt', '')}_{meta}_{constr}_{ls}.txt"
                    output_path = os.path.join(OUTPUT_DIR, output_file)
                    
                    strategy = f"{meta}+{constr}+{ls}"
                    
                    cmd = [BINARY_PATH, input_path, output_path, strategy]
                    
                    # Display
                    desc = f"{meta} + {constr} + {ls}"
                    print(f"{desc.ljust(50)} ... ", end="", flush=True)
                    
                    start_time = time.time()
                    try:
                        # Timeout 20s for metaheuristics
                        result = subprocess.run(
                            cmd, 
                            capture_output=True, 
                            text=True, 
                            timeout=20 
                        )
                        
                        duration = time.time() - start_time
                        
                        if result.returncode == 0:
                            print(f"{GREEN}OK{RESET} ({duration:.4f}s)")
                        else:
                            print(f"{RED}ERROR{RESET} (Exit Code: {result.returncode})")
                            failed_tests.append((instance, strategy, "Runtime Error", result.stderr))
                            
                    except subprocess.TimeoutExpired:
                        print(f"{RED}TIMEOUT{RESET} (>20s)")
                        failed_tests.append((instance, strategy, "Infinite Loop/Timeout", ""))
                    except Exception as e:
                        print(f"{RED}EXCEPTION{RESET}")
                        print(f"  {str(e)}")
                        failed_tests.append((instance, strategy, "Exception", str(e)))

    print("-" * 80)
    if failed_tests:
        print(f"{RED}Tests failed: {len(failed_tests)}{RESET}")
        for inst, strat, reason, details in failed_tests:
            print(f"  - {inst} [{strat}]: {reason}")
            if details:
                lines = details.strip().split('\n')
                print(f"    Stderr: ... {lines[-1] if lines else ''}") 
    else:
        print(f"{GREEN}All combinations passed successfully!{RESET}")

if __name__ == "__main__":
    if not os.path.exists(INPUT_DIR):
        print(f"{RED}Error: Input directory '{INPUT_DIR}' not found.{RESET}")
        sys.exit(1)
    run_test()
