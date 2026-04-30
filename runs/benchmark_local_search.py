import os
import subprocess
import time
import csv
import re

# Configurações
BINARY_PATH = "./target/release/spo-ml-rust"
DATASETS_DIR = "datasets"
BEST_VAL_PATH = "best_solutions/best_objectives.csv"
OUTPUT_CSV = "outputs/benchmark_local_search_report.csv"
TEMP_OUTPUT = "outputs/temp_ls_sol.txt"

# Definidos com base no benchmark anterior
BEST_CONSTR = "aisle_adaptive"
WORST_CONSTR = "static"

LOCAL_SEARCHES = [
    "none",
    "swap", "a_swap",
    "vnd", "a_vnd",
    "hvnd",
    "tabu", "a_tabu",
    "lahc", "a_lahc"
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

def get_score_from_output(stdout, pattern):
    if pattern in stdout:
        score_match = re.search(pattern + r":\s*([\d\.]+)", stdout)
        if score_match:
            return float(score_match.group(1))
    return 0.0

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

    print(f"Iniciando benchmark de {len(LOCAL_SEARCHES)} buscas locais...")
    print(f"Usando Construtivas: {BEST_CONSTR} (Melhor) e {WORST_CONSTR} (Pior)")
    print(f"O relatório será salvo em: {OUTPUT_CSV}")
    print("-" * 120)

    for ds, inst_file in instances:
        instance_key = f"{ds}/{inst_file}"
        input_path = os.path.join(DATASETS_DIR, ds, inst_file)
        best_known = best_values.get(instance_key, 0.0)

        for constr in [BEST_CONSTR, WORST_CONSTR]:
            constr_label = "BEST" if constr == BEST_CONSTR else "WORST"
            
            for ls in LOCAL_SEARCHES:
                strategy = f"single+{constr}+{ls}"
                cmd = [BINARY_PATH, input_path, TEMP_OUTPUT, strategy, "42"]
                
                start_time = time.time()
                try:
                    process = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                    duration = time.time() - start_time
                    stdout = process.stdout
                    
                    initial_score = 0.0
                    final_score = 0.0
                    status = "OK"
                    improved = "false"
                    
                    if process.returncode == 0:
                        initial_score = get_score_from_output(stdout, "Initial Score")
                        final_score = get_score_from_output(stdout, "Final Score")
                        
                        if "Improved: true" in stdout:
                            improved = "true"
                        elif "Solution NOT Feasible!" in stdout:
                            status = "INFEASIBLE"
                    else:
                        status = f"CRASH ({process.returncode})"

                    gap = 0.0
                    if best_known > 0:
                        gap = ((best_known - final_score) / best_known) * 100

                    diff = final_score - initial_score
                    
                    results.append({
                        "dataset": ds,
                        "instance": inst_file,
                        "constructive": constr,
                        "constr_quality": constr_label,
                        "local_search": ls,
                        "initial_score": f"{initial_score:.4f}",
                        "final_score": f"{final_score:.4f}",
                        "improvement": f"{diff:.4f}",
                        "best_known": f"{best_known:.4f}",
                        "gap_percent": f"{gap:.2f}%",
                        "improved": improved,
                        "time_seconds": f"{duration:.4f}",
                        "status": status
                    })
                    
                    improved_flag = "[*]" if improved == "true" else "[ ]"
                    print(f"[{status}] {inst_file.ljust(18)} | {constr_label.ljust(5)} + {ls.ljust(6)} {improved_flag} | Initial: {initial_score:>8.2f} | Final: {final_score:>8.2f} | Gain: {diff:>7.2f} | Gap: {gap:>6.2f}% | Time: {duration:.3f}s")

                except subprocess.TimeoutExpired:
                    print(f"[TIMEOUT] {inst_file.ljust(18)} | {constr_label.ljust(5)} + {ls.ljust(6)}")
                    results.append({
                        "dataset": ds, "instance": inst_file, "constructive": constr, "constr_quality": constr_label,
                        "local_search": ls, "initial_score": "0.0", "final_score": "0.0", 
                        "improvement": "0.0", "best_known": f"{best_known:.4f}",
                        "gap_percent": "100%", "improved": "N/A", "time_seconds": "120.0", "status": "TIMEOUT"
                    })

    # Salvar resultados no CSV
    with open(OUTPUT_CSV, mode='w', newline='') as f:
        fieldnames = ["dataset", "instance", "constructive", "constr_quality", "local_search", "initial_score", "final_score", "improvement", "best_known", "gap_percent", "improved", "time_seconds", "status"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("-" * 120)
    print(f"Benchmark concluído! CSV gerado em {OUTPUT_CSV}")

if __name__ == "__main__":
    run_benchmark()
