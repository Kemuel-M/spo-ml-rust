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

HEURISTICS = [
    "static",
    "adaptive",
    "random",
    "aisle_static",
    "aisle_adaptive",
    "aisle_random"
]

BINARY_PATH = "./target/release/spo-ml-rust"
INPUT_DIR = "datasets/a"
OUTPUT_DIR = "outputs/test_run"

# Colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def run_test():
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Starting tests on {len(INSTANCES)} instances with {len(HEURISTICS)} heuristics.")
    print("-" * 60)

    failed_tests = []
    
    for instance in INSTANCES:
        for heuristic in HEURISTICS:
            input_path = os.path.join(INPUT_DIR, instance)
            output_file = f"{instance.replace('.txt', '')}_{heuristic}.txt"
            output_path = os.path.join(OUTPUT_DIR, output_file)
            
            # Construct strategy string: single phase + constructive + no local search
            strategy = f"single+{heuristic}+none"
            
            cmd = [BINARY_PATH, input_path, output_path, strategy]
            
            print(f"Testing {instance} with {heuristic.ljust(15)} ... ", end="", flush=True)
            
            start_time = time.time()
            try:
                # Run with timeout to catch infinite loops
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=5 # 5 seconds is generous for constructive
                )
                
                duration = time.time() - start_time
                
                if result.returncode == 0:
                    print(f"{GREEN}OK{RESET} ({duration:.4f}s)")
                else:
                    print(f"{RED}ERROR{RESET} (Exit Code: {result.returncode})")
                    print(f"  Stderr: {result.stderr.strip()}")
                    failed_tests.append((instance, heuristic, "Runtime Error", result.stderr))
                    
            except subprocess.TimeoutExpired:
                print(f"{RED}TIMEOUT{RESET} (>5s)")
                failed_tests.append((instance, heuristic, "Infinite Loop/Timeout", ""))
            except Exception as e:
                print(f"{RED}EXCEPTION{RESET}")
                print(f"  {str(e)}")
                failed_tests.append((instance, heuristic, "Exception", str(e)))

    print("-" * 60)
    if failed_tests:
        print(f"{RED}Some tests failed:{RESET}")
        for inst, heur, reason, details in failed_tests:
            print(f"  - {inst} [{heur}]: {reason}")
            if details:
                print(f"    Details: {details[:200]}...") # Truncate long error messages
    else:
        print(f"{GREEN}All tests passed successfully!{RESET}")

if __name__ == "__main__":
    if not os.path.exists(INPUT_DIR):
        print(f"{RED}Error: Input directory '{INPUT_DIR}' not found.{RESET}")
        print("Please run this script from the 'spo-ml-rust' root.")
        sys.exit(1)
        
    run_test()
