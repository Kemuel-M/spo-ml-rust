import os
import subprocess
import time
import csv
import re

# Configurações
BINARY_PATH = "./target/release/spo-ml-rust"
DATASETS_DIR = "datasets"
BEST_VAL_PATH = "best_solutions/best_objectives.csv"
OUTPUT_CSV = "outputs/benchmark_constructives_report.csv"
TEMP_OUTPUT = "outputs/temp_sol.txt"

HEURISTICS = [
    "static",
    "adaptive",
    "random",
    "aisle",
    "aisle_adaptive",
    "aisle_random"
]

def load_best_values():
    best_vals = {}
    if os.path.exists(BEST_VAL_PATH):
        with open(BEST_VAL_PATH, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = f"{row['dataset']}/{row['instance']}"
                best_vals[key] = float(row['best_objective'])
    return best_vals

def run_benchmark():
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    best_values = load_best_values()
    results = []

    # Localizar todas as instâncias nos subdiretórios a
    instances = []
    for ds in ['a']:
        ds_path = os.path.join(DATASETS_DIR, ds)
        if os.path.exists(ds_path):
            for file in sorted(os.listdir(ds_path)):
                if file.endswith(".txt"):
                    instances.append((ds, file))

    print(f"Iniciando benchmark de {len(HEURISTICS)} heurísticas em {len(instances)} instâncias...")
    print(f"O relatório será salvo em: {OUTPUT_CSV}")
    print("-" * 70)

    for ds, inst_file in instances:
        instance_key = f"{ds}/{inst_file}"
        input_path = os.path.join(DATASETS_DIR, ds, inst_file)
        best_known = best_values.get(instance_key, 0.0)

        for heur in HEURISTICS:
            # Comando: single shot + heurística + sem busca local
            strategy = f"single+{heur}+none"
            cmd = [BINARY_PATH, input_path, TEMP_OUTPUT, strategy, "42"]
            
            start_time = time.time()
            try:
                process = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                duration = time.time() - start_time
                stdout = process.stdout
                
                score = 0.0
                status = "ERROR"
                
                if process.returncode == 0:
                    if "Final Score:" in stdout:
                        score_match = re.search(r"Final Score:\s*([\d\.]+)", stdout)
                        if score_match:
                            score = float(score_match.group(1))
                            status = "OK"
                    elif "Solution NOT Feasible!" in stdout:
                        status = "INFEASIBLE"
                else:
                    status = f"CRASH ({process.returncode})"

                gap = 0.0
                if best_known > 0:
                    gap = ((best_known - score) / best_known) * 100

                results.append({
                    "dataset": ds,
                    "instance": inst_file,
                    "heuristic": heur,
                    "found_score": f"{score:.4f}",
                    "best_known": f"{best_known:.4f}",
                    "gap_percent": f"{gap:.2f}%",
                    "time_seconds": f"{duration:.4f}",
                    "status": status
                })
                
                print(f"[{status}] {instance_key.ljust(20)} | {heur.ljust(15)} | Score: {score:>8.2f} | Gap: {gap:>6.2f}%")

            except subprocess.TimeoutExpired:
                print(f"[TIMEOUT] {instance_key.ljust(20)} | {heur.ljust(15)}")
                results.append({
                    "dataset": ds, "instance": inst_file, "heuristic": heur,
                    "found_score": "0.0", "best_known": f"{best_known:.4f}",
                    "gap_percent": "100%", "time_seconds": "30.0", "status": "TIMEOUT"
                })
            except Exception as e:
                print(f"[EXCEPTION] {heur}: {str(e)}")

    # Salvar resultados no CSV
    with open(OUTPUT_CSV, mode='w', newline='') as f:
        fieldnames = ["dataset", "instance", "heuristic", "found_score", "best_known", "gap_percent", "time_seconds", "status"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("-" * 70)
    print(f"Benchmark concluído! CSV gerado em {OUTPUT_CSV}")

if __name__ == "__main__":
    run_benchmark()
