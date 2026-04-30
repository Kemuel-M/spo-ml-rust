import os
import subprocess
import time
import csv
import re

# Configurações
BINARY_PATH = "./target/release/spo-ml-rust"
DATASETS_DIR = "datasets"
BEST_VAL_PATH = "best_solutions/best_objectives.csv"
OUTPUT_CSV = "outputs/benchmark_metaheuristics_report.csv"
TEMP_OUTPUT = "outputs/temp_meta_sol.txt"
CHECKER_PATH = "runs/checker.py"

# Definições de Metaheurísticas
# (nome_base, prefixo_corredor, descricao)
METAS = [
    ("grasp:50", "grasp:50", "GRASP"),
    ("ils:30", "a_ils:30", "ILS"),
    ("ga:50", "a_ga:50", "GA"),
    ("bpso:50", "a_bpso:50", "BPSO"),
    ("sa", "a_sa", "SA"),
]

# Definições de Busca Local
# (nome_base, prefixo_corredor, descricao)
LOCAL_SEARCHES = [
    ("none", "none", "None"),
    ("swap", "a_swap", "Swap"),
    ("vnd", "a_vnd", "VND"),
    ("tabu", "a_tabu", "Tabu"),
    ("lahc", "a_lahc", "LAHC"),
    ("hvnd", "hvnd", "HVND"),
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

def validate_solution(input_path, output_path):
    if not os.path.exists(output_path):
        return False, 0.0, "MISSING_OUTPUT"
    try:
        cmd = ["python3", CHECKER_PATH, input_path, output_path]
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
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
    if not os.path.exists("outputs"): os.makedirs("outputs")
    best_values = load_best_values()
    results = []

    # Instâncias selecionadas para o benchmark (todas do dataset 'a')
    instances = []
    for ds in ['a']:
        ds_path = os.path.join(DATASETS_DIR, ds)
        if os.path.exists(ds_path):
            all_files = sorted([f for f in os.listdir(ds_path) if f.endswith(".txt")])
            for file in all_files:
                instances.append((ds, file))

    print(f"Iniciando Benchmark Extensivo de Metaheurísticas...")
    print(f"Instâncias: {len(instances)} | Metas: {len(METAS)} | LS: {len(LOCAL_SEARCHES)}")
    print(f"O relatório será salvo em: {OUTPUT_CSV}")
    print("-" * 150)

    for ds, inst_file in instances:
        instance_key = f"{ds}/{inst_file}"
        input_path = os.path.join(DATASETS_DIR, ds, inst_file)
        best_known = best_values.get(instance_key, 0.0)

        for meta_base, meta_aisle, meta_desc in METAS:
            for dim in ["Orders", "Aisles"]:
                # Seleciona o prefixo da meta e da busca local baseado na dimensão
                m_cfg = meta_base if dim == "Orders" else meta_aisle
                constr = "random:0.1" if dim == "Orders" else "aisle_random:0.1"
                
                for ls_base, ls_aisle, ls_desc in LOCAL_SEARCHES:
                    # BPSO Pure (None) já está no loop, mas garantimos que as outras combinações ocorram
                    ls_cfg = ls_base if dim == "Orders" else ls_aisle
                    
                    strategy = f"{m_cfg}+{constr}+{ls_cfg}"
                    desc_completa = f"{meta_desc} ({dim}) + {ls_desc}"
                    
                    cmd = [BINARY_PATH, input_path, TEMP_OUTPUT, strategy, "42"]
                    
                    start_time = time.time()
                    try:
                        # Timeout de 150 segundos por execução (GA pode demorar mais)
                        process = subprocess.run(cmd, capture_output=True, text=True, timeout=150)
                        duration = time.time() - start_time
                        
                        status = "OK"
                        if process.returncode != 0:
                            status = f"CRASH ({process.returncode})"
                            final_score = 0.0
                        else:
                            is_valid, final_score, checker_status = validate_solution(input_path, TEMP_OUTPUT)
                            if not is_valid: status = checker_status

                        gap = 0.0
                        if best_known > 0:
                            gap = ((best_known - final_score) / best_known) * 100

                        results.append({
                            "instance": instance_key,
                            "construtive": constr,
                            "local_search": ls_cfg,
                            "metaheuristica": desc_completa,
                            "score": f"{final_score:.4f}",
                            "best_known": f"{best_known:.4f}",
                            "gap_percent": f"{gap:.2f}%",
                            "time_seconds": f"{duration:.4f}",
                            "status": status
                        })
                        
                        print(f"[{status.ljust(10)}] {inst_file.ljust(18)} | {desc_completa.ljust(30)} | Score: {final_score:>8.2f} | Gap: {gap:>6.2f}% | Time: {duration:.2f}s")

                    except subprocess.TimeoutExpired:
                        print(f"[TIMEOUT   ] {inst_file.ljust(18)} | {desc_completa.ljust(30)}")
                        results.append({
                            "instance": instance_key, "construtive": constr, "local_search": ls_cfg,
                            "metaheuristica": desc_completa, "score": "0.0000", "best_known": f"{best_known:.4f}",
                            "gap_percent": "100.00%", "time_seconds": "150.00", "status": "TIMEOUT"
                        })

    # Salvar resultados no CSV
    with open(OUTPUT_CSV, mode='w', newline='') as f:
        fieldnames = ["instance", "construtive", "local_search", "metaheuristica", "score", "best_known", "gap_percent", "time_seconds", "status"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("-" * 150)
    print(f"Benchmark concluído! CSV gerado em {OUTPUT_CSV}")

if __name__ == "__main__":
    run_benchmark()
