import os
import subprocess
import time
import csv
import re

# =============================================================================
# CONFIGURAÇÕES EDITÁVEIS
# =============================================================================

# Dataset e Execução
DATASET = "a"           # Opções: "a", "b", "x"
SEED = "0"             # Semente aleatória (use "0" para semente do sistema)
TIMEOUT_LIMIT = 600     # Tempo máximo por instância em segundos (Python)
MAX_INSTANCES = 0       # Limite de instâncias a processar (0 para todas)
RUNS_PER_INSTANCE = 1   # Reduzi para 5 runs por padrão pois BPSO é mais pesado que GRASP

# Parâmetros da Meta-heurística (Hybrid BPSO)
ITERATIONS = 1000        # Número de iterações do BPSO
CONSTRUCTIVE = "adaptive" # Estratégia base para população inicial
LOCAL_SEARCH = "hvnd"   # Estratégia de busca local (memético)

# Caminhos de Arquivos
BINARY_PATH = "./target/release/spo-ml-rust"
BEST_VAL_PATH = "best_solutions/best_objectives.csv"
CHECKER_PATH = "runs/checker.py"
OUTPUT_DIR = "outputs"

# =============================================================================
# DERIVADOS
# =============================================================================

STRATEGY = f"h_bpso:{ITERATIONS}+{CONSTRUCTIVE}+{LOCAL_SEARCH}"
STRATEGY_DESC = f"H-BPSO-{ITERATIONS} ({CONSTRUCTIVE} + {LOCAL_SEARCH})"

DATASETS_DIR = f"datasets/{DATASET}"
OUTPUT_CSV = f"{OUTPUT_DIR}/benchmark_h_bpso_{LOCAL_SEARCH}_{DATASET}.csv"

def load_best_values():
    best_vals = {}
    if os.path.exists(BEST_VAL_PATH):
        with open(BEST_VAL_PATH, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['dataset'] == DATASET:
                    key = row['instance']
                    best_vals[key] = float(row['best_objective'])
    return best_vals

def validate_solution(input_path, output_path):
    if not os.path.exists(output_path):
        return False, 0.0, "MISSING_OUTPUT"
    try:
        cmd = ["python3", CHECKER_PATH, input_path, output_path]
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        stdout = process.stdout
        is_feasible = "Is solution feasible: True" in stdout
        score = 0.0
        if is_feasible:
            match = re.search(r"Objective function value: ([\d\.]+)", stdout)
            if match: score = float(match.group(1))
            return True, score, "OK"
        else:
            return False, 0.0, "INFEASIBLE"
    except Exception as e:
        return False, 0.0, f"CHECKER_ERROR ({str(e)})"

def run_benchmark():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    best_values = load_best_values()
    
    if not os.path.exists(DATASETS_DIR):
        print(f"Erro: Diretorio {DATASETS_DIR} nao encontrado.")
        return

    instances = sorted([f for f in os.listdir(DATASETS_DIR) if f.endswith(".txt")])
    if MAX_INSTANCES > 0 and len(instances) > MAX_INSTANCES:
        instances = instances[:MAX_INSTANCES]

    print(f"Iniciando Benchmark [{DATASET.upper()}]: {STRATEGY_DESC}")
    print(f"Estratégia: {STRATEGY}")
    print(f"Instâncias: {len(instances)}")
    print(f"Rodadas por Instância: {RUNS_PER_INSTANCE}")
    print(f"Relatório: {OUTPUT_CSV}")
    print("-" * 100)

    fieldnames = [
        "instance", "run", "strategy", "score", "total_iterations", 
        "final_avg_obj", "final_avg_v", "ls_hit_rate_final", "imp_rate_final",
        "orders_split", "aisles_split",
        "best_known", "gap_percent", "time_seconds", "status"
    ]
    
    with open(OUTPUT_CSV, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for inst_file in instances:
            input_path = os.path.join(DATASETS_DIR, inst_file)
            best_known = best_values.get(inst_file, 0.0)
            
            print(f"Processando {inst_file} ({RUNS_PER_INSTANCE} runs):")
            
            for run_idx in range(1, RUNS_PER_INSTANCE + 1):
                current_seed = str(int(SEED) + run_idx) if SEED != "0" else "0"
                temp_output = os.path.join(OUTPUT_DIR, f"temp_bench_hbpso_{DATASET}_{inst_file}_r{run_idx}")
                
                print(f"  Run {run_idx:02d}/{RUNS_PER_INSTANCE:02d}...", end=" ", flush=True)
                
                start_time = time.time()
                try:
                    cmd = [BINARY_PATH, input_path, temp_output, STRATEGY, current_seed]
                    process = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_LIMIT)
                    elapsed = time.time() - start_time
                    stdout = process.stdout

                    # Extrair total de iterações
                    iter_match = re.search(r"\[BPSO\] Finished in (\d+) iterations", stdout)
                    total_iters = iter_match.group(1) if iter_match else ""

                    # Extrair Hybrid Split
                    hybrid_match = re.search(r"\[Hybrid Mode: (\d+) Orders / (\d+) Aisles\]", stdout)
                    o_split = hybrid_match.group(1) if hybrid_match else ""
                    a_split = hybrid_match.group(2) if hybrid_match else ""

                    # Extrair métricas da última linha de log
                    # Regex para capturar: Avg Obj | Imp% | Stag | w | Turb | AvgV | LSHit
                    log_lines = re.findall(r"^\s+\d+\s+\|\s+[\d\.]+\s+\|\s+([\d\.]+)\s+\|\s+([\d\.]+)%\s+\|\s+\d+\s+\|\s+[\d\.]+\s+\|\s+[\d\.]+\s+\|\s+([\d\.]+)\s+\|\s+([\d\.]+)%\s+\|\s+\d+s", stdout, re.MULTILINE)
                    
                    final_avg_obj = ""
                    final_imp_rate = ""
                    final_avg_v = ""
                    final_ls_hit = ""

                    if log_lines:
                        last_log = log_lines[-1]
                        final_avg_obj = last_log[0]
                        final_imp_rate = last_log[1]
                        final_avg_v = last_log[2]
                        final_ls_hit = last_log[3]

                    if process.returncode != 0:
                        print(f"FALHA (code {process.returncode})")
                        writer.writerow({
                            "instance": inst_file, "run": run_idx, "strategy": STRATEGY,
                            "score": 0.0, "total_iterations": total_iters, "best_known": best_known, "gap_percent": 100.0,
                            "time_seconds": elapsed, "status": f"ERROR_{process.returncode}"
                        })
                        continue

                    is_feasible, score, status = validate_solution(input_path, temp_output)
                    
                    gap = 0.0
                    if best_known > 0:
                        gap = ((best_known - score) / best_known) * 100.0
                    
                    print(f"Score: {score:.4f} | Iters: {total_iters} | AvgV: {final_avg_v} | LSHit: {final_ls_hit}% | Gap: {gap:.2f}%")
                    
                    row = {
                        "instance": inst_file, "run": run_idx, "strategy": STRATEGY,
                        "score": score, 
                        "total_iterations": total_iters,
                        "final_avg_obj": final_avg_obj,
                        "final_avg_v": final_avg_v,
                        "ls_hit_rate_final": final_ls_hit,
                        "imp_rate_final": final_imp_rate,
                        "orders_split": o_split,
                        "aisles_split": a_split,
                        "best_known": best_known, 
                        "gap_percent": gap,
                        "time_seconds": elapsed, 
                        "status": status
                    }
                    writer.writerow(row)
                    csvfile.flush()
                    
                    if os.path.exists(temp_output):
                        os.remove(temp_output)
                        
                except subprocess.TimeoutExpired:
                    print("TIMEOUT")
                    writer.writerow({
                        "instance": inst_file, "run": run_idx, "strategy": STRATEGY,
                        "score": 0.0, "best_known": best_known, "gap_percent": 100.0,
                        "time_seconds": float(TIMEOUT_LIMIT), "status": "TIMEOUT"
                    })
                except Exception as e:
                    print(f"ERRO: {str(e)}")
                    writer.writerow({
                        "instance": inst_file, "run": run_idx, "strategy": STRATEGY,
                        "score": 0.0, "best_known": best_known, "gap_percent": 100.0,
                        "time_seconds": 0.0, "status": f"EXCEPTION_{type(e).__name__}"
                    })

    print("-" * 100)
    print(f"Benchmark concluído. Resultados salvos em {OUTPUT_CSV}")

if __name__ == "__main__":
    run_benchmark()
