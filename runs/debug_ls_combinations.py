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

CONSTRUCTIVES = [
    "static",
    "adaptive",
    "random",
    "aisle_static",
    "aisle_adaptive",
    "aisle_random"
]

LOCAL_SEARCHES = [
    "vnd",
    "tabu",
    "lahc",
    "swap",
    "insert",
    "remove"
]

BINARY_PATH = "./target/release/spo-ml-rust"
INPUT_DIR = "datasets/a"
OUTPUT_DIR = "outputs/test_ls_run"

# Colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BLUE = "\033[94m"

def run_test():
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    total_tests = len(INSTANCES) * len(CONSTRUCTIVES) * len(LOCAL_SEARCHES)
    print(f"Starting comprehensive tests: {len(INSTANCES)} Instances x {len(CONSTRUCTIVES)} Constructives x {len(LOCAL_SEARCHES)} LS = {total_tests} runs.")
    print("-" * 80)

    failed_tests = []
    current_count = 0
    
    for instance in INSTANCES:
        print(f"\n{BLUE}Instance: {instance}{RESET}")
        for constr in CONSTRUCTIVES:
            for ls in LOCAL_SEARCHES:
                current_count += 1
                
                input_path = os.path.join(INPUT_DIR, instance)
                output_file = f"{instance.replace('.txt', '')}_{constr}_{ls}.txt"
                output_path = os.path.join(OUTPUT_DIR, output_file)
                
                # Strategy: single phase + constructive + local search
                strategy = f"single+{constr}+{ls}"
                
                cmd = [BINARY_PATH, input_path, output_path, strategy]
                
                # Format string for alignment
                test_desc = f"[{current_count}/{total_tests}] {constr} + {ls}"
                print(f"{test_desc.ljust(45)} ... ", end="", flush=True)
                
                start_time = time.time()
                try:
                    # Run with timeout (increased to 10s for heavier LS like Tabu/LAHC)
                    result = subprocess.run(
                        cmd, 
                        capture_output=True, 
                        text=True, 
                        timeout=10 
                    )
                    
                    duration = time.time() - start_time
                    
                    if result.returncode == 0:
                        print(f"{GREEN}OK{RESET} ({duration:.4f}s)")
                    else:
                        print(f"{RED}ERROR{RESET} (Exit Code: {result.returncode})")
                        # print(f"  Stderr: {result.stderr.strip()}")
                        failed_tests.append((instance, constr, ls, "Runtime Error", result.stderr))
                        
                except subprocess.TimeoutExpired:
                    print(f"{RED}TIMEOUT{RESET} (>10s)")
                    failed_tests.append((instance, constr, ls, "Infinite Loop/Timeout", ""))
                except Exception as e:
                    print(f"{RED}EXCEPTION{RESET}")
                    print(f"  {str(e)}")
                    failed_tests.append((instance, constr, ls, "Exception", str(e)))

    print("-" * 80)
    if failed_tests:
        print(f"{RED}Tests failed: {len(failed_tests)}{RESET}")
        for inst, constr, ls, reason, details in failed_tests:
            print(f"  - {inst} [{constr} + {ls}]: {reason}")
            if details:
                # Print only the last few lines of stderr to keep it readable
                lines = details.strip().split('\n')
                last_lines = "\n".join(lines[-3:])
                print(f"    Stderr: ...\n{last_lines}") 
    else:
        print(f"{GREEN}All {total_tests} combinations passed successfully!{RESET}")

if __name__ == "__main__":
    if not os.path.exists(INPUT_DIR):
        print(f"{RED}Error: Input directory '{INPUT_DIR}' not found.{RESET}")
        print("Please run this script from the 'spo-ml-rust' root.")
        sys.exit(1)
        
    run_test()
