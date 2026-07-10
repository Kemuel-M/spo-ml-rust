import os
import subprocess
import time
import csv
import re

# ==========================================
# CONFIGURAÇÕES DO BENCHMARK
# ==========================================
BINARY_PATH = "./target/release/spo-ml-rust"
DATASETS_DIR = "datasets"
BEST_VAL_PATH = "best_solutions/best_objectives.csv"
OUTPUT_CSV = "outputs/benchmark_local_search_report.csv"
TEMP_OUTPUT = "outputs/temp_ls_sol.txt"

# Datasets a serem processados (ex: ['a', 'b', 'x'])
DATASETS = ['a', 'b', 'x']

# Construtivos Base para Inicializar a Busca Local
# Usamos o 'oa' (Order Adaptive) por padrão, pois é o mediano perfeito
CONSTRUCTIVES = ["oa", "aa"]

# Estratégias de Busca Local para Avaliar
# A string 'none' é usada como baseline (apenas o resultado do construtivo puro)
LOCAL_SEARCHES = [
    "swap",
    "a_swap",
    "tabu",
    "a_tabu",
    "lahc",
    "a_lahc",
    "vnd",
    "a_vnd",
    "hvnd"
]


# Semente fixa para reprodutibilidade dos benchmarks determinísticos
# Se for "0", usará semente do sistema (aleatória)
SEED = "42"

# Quantidade de vezes que cada instância será executada (útil se construtivo ou LS for estocástico)
RUNS_PER_INSTANCE = 10
# ==========================================

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

    # Localizar todas as instâncias nos subdiretórios configurados
    instances = []
    for ds in DATASETS:
        ds_path = os.path.join(DATASETS_DIR, ds)
        if os.path.exists(ds_path):
            for file in sorted(os.listdir(ds_path)):
                if file.endswith(".txt"):
                    instances.append((ds, file))

    total_runs = len(instances) * len(CONSTRUCTIVES) * len(LOCAL_SEARCHES) * RUNS_PER_INSTANCE
    print(f"Iniciando benchmark de Buscas Locais...")
    print(f"Total de execuções planejadas: {total_runs}")
    print(f"O relatório será salvo em: {OUTPUT_CSV}")
    print("-" * 140)

    for ds, inst_file in instances:
        instance_key = f"{ds}/{inst_file}"
        input_path = os.path.join(DATASETS_DIR, ds, inst_file)
        best_known = best_values.get(instance_key, 0.0)

        for constr in CONSTRUCTIVES:
            for ls in LOCAL_SEARCHES:
                # Modificador para chamar a busca local de "tiro unico" (apenas 1 vez após construção)
                strategy = f"single+{constr}+{ls}"
                
                for run_idx in range(1, RUNS_PER_INSTANCE + 1):
                    current_seed = str(int(SEED) + run_idx) if SEED != "0" else "0"
                    cmd = [BINARY_PATH, input_path, TEMP_OUTPUT, strategy, current_seed]
                    
                    start_time = time.time()
                    try:
                        process = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                        duration = time.time() - start_time
                        stdout = process.stdout
                        
                        initial_score = 0.0
                        final_score = 0.0
                        status = "OK"
                        
                        if process.returncode == 0:
                            initial_score = get_score_from_output(stdout, "Initial Score")
                            final_score = get_score_from_output(stdout, "Final Score")
                            
                            if "Solution NOT Feasible!" in stdout:
                                status = "INFEASIBLE"
                                
                            # Tratamento para caso a busca local seja None, o inicial é igual ao final
                            if initial_score == 0.0 and final_score > 0.0:
                                initial_score = final_score
                        else:
                            status = f"CRASH ({process.returncode})"

                        gap = 0.0
                        if best_known > 0 and final_score > 0:
                            gap = ((best_known - final_score) / best_known) * 100

                        # Diferença direta em pontos absolutos (initial - final, pois queremos maximizar ou...
                        # O score é Itens/Corredor (maior é melhor), então ganho = final - initial
                        gain = final_score - initial_score
                        improved = "true" if gain > 0.0001 else "false"
                        
                        results.append({
                            "dataset": ds,
                            "instance": inst_file,
                            "run": run_idx,
                            "constructive": constr,
                            "local_search": ls,
                            "initial_score": f"{initial_score:.4f}",
                            "final_score": f"{final_score:.4f}",
                            "gain_score": f"{gain:.4f}",
                            "best_known": f"{best_known:.4f}",
                            "gap_percent": f"{gap:.2f}%",
                            "improved": improved,
                            "time_seconds": f"{duration:.4f}",
                            "status": status
                        })
                        
                        improved_flag = "[+]" if improved == "true" else "[ ]"
                        print(f"[{status}] {inst_file.ljust(18)} | {constr.ljust(5)} + {ls.ljust(6)} | Run: {run_idx:02d} {improved_flag} | Initial: {initial_score:>8.2f} | Final: {final_score:>8.2f} | Gain: {gain:>7.2f} | Gap: {gap:>6.2f}% | Time: {duration:.3f}s")

                    except subprocess.TimeoutExpired:
                        print(f"[TIMEOUT] {inst_file.ljust(18)} | {constr.ljust(5)} + {ls.ljust(6)} | Run: {run_idx:02d}")
                        results.append({
                            "dataset": ds, "instance": inst_file, "run": run_idx, "constructive": constr,
                            "local_search": ls, "initial_score": "0.0", "final_score": "0.0", 
                            "gain_score": "0.0", "best_known": f"{best_known:.4f}",
                            "gap_percent": "100%", "improved": "N/A", "time_seconds": "120.0", "status": "TIMEOUT"
                        })

    # Salvar resultados no CSV
    with open(OUTPUT_CSV, mode='w', newline='') as f:
        fieldnames = ["dataset", "instance", "run", "constructive", "local_search", "initial_score", "final_score", "gain_score", "best_known", "gap_percent", "improved", "time_seconds", "status"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("-" * 140)
    print(f"Benchmark concluído! CSV gerado em {OUTPUT_CSV}")

if __name__ == "__main__":
    run_benchmark()
