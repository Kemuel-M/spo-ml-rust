import os
import subprocess
import time
import csv
import re

# =============================================================================
# CONFIGURAÇÕES EDITÁVEIS
# =============================================================================

# --- Dataset e Execução ---
DATASETS = ["a", "b", "x"]            # Pode ser uma string ex: "a" ou uma lista ex: ["a", "b", "x"]

# Lista de instâncias específicas para rodar. Se a lista não estiver vazia, rodará apenas estas instâncias.
# Deixe vazia (SELECTED_INSTANCES = []) para rodar todas as instâncias do dataset (sujeito ao MAX_INSTANCES).
# Exemplo: SELECTED_INSTANCES = ["instance_0014.txt", "instance_0015.txt"]
SELECTED_INSTANCES = []

SEED = "42"                  # Semente aleatória (use "0" para semente do sistema)
TIMEOUT_LIMIT = 30          # Tempo máximo por instância em segundos

# Limite de instâncias a processar (0 para processar todas do diretório).
# Nota: Este parâmetro é ignorado caso SELECTED_INSTANCES não esteja vazia.
MAX_INSTANCES = 0           

RUNS_PER_INSTANCE = 10      # Quantidade de vezes que cada instância será executada

# --- Estratégias Construtivas ---
# Pode ser uma string única ou uma lista de strings. O script testará todas as combinações.
# Todos os algoritmos construtivos por padrão na lista:
CONSTRUCTIVES = [
    "order_static",
    "order_adaptive",
    "order_random",
    "aisle_static",
    "aisle_adaptive",
    "aisle_random"
]

# --- Caminhos de Arquivos ---
BINARY_PATH = "./target/release/spo-ml-rust"
BEST_VAL_PATH = "best_solutions/best_objectives.csv"
CHECKER_PATH = "runs/checker.py"
OUTPUT_DIR = "outputs"

# =============================================================================
# DERIVADOS
# =============================================================================

# Não temos um DATASETS_DIR global mais, pois vamos iterar sobre a lista.


def load_best_values(datasets_list):
    best_vals = {}
    if os.path.exists(BEST_VAL_PATH):
        with open(BEST_VAL_PATH, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['dataset'] in datasets_list:
                    key = f"{row['dataset']}/{row['instance']}"
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
    if not os.path.exists(OUTPUT_DIR): 
        os.makedirs(OUTPUT_DIR)
        
    datasets_list = DATASETS if isinstance(DATASETS, list) else [DATASETS]
    best_values = load_best_values(datasets_list)
    
    # Garante que sejam listas mesmo se o usuário configurar apenas uma string
    constructives_list = CONSTRUCTIVES if isinstance(CONSTRUCTIVES, list) else [CONSTRUCTIVES]

    print(f"Iniciando Benchmark de Construtivos nos Datasets: {', '.join(datasets_list)}")
    print(f"Construtivos: {', '.join(constructives_list)}")
    print(f"Rodadas por Instância: {RUNS_PER_INSTANCE}")
    print("-" * 100)

    fieldnames = [
        "dataset", "instance", "run", "strategy", "constructive", 
        "score", "best_known", "gap_percent", "time_seconds", "status"
    ]
    
    # Criar um nome detalhado com todas as configurações utilizadas
    constructives_str = "-".join(constructives_list)
    datasets_str = "-".join(datasets_list)
    suffix = "_selected" if SELECTED_INSTANCES else ""
    
    base_csv_name = f"bench_constructives_ds-[{datasets_str}]_t{TIMEOUT_LIMIT}_c-[{constructives_str}]{suffix}"
    
    # Lógica para não sobrescrever caso o arquivo já exista
    counter = 1
    output_csv = f"{OUTPUT_DIR}/{base_csv_name}.csv"
    while os.path.exists(output_csv):
        output_csv = f"{OUTPUT_DIR}/{base_csv_name}_{counter}.csv"
        counter += 1

    with open(output_csv, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for current_dataset in datasets_list:
            dataset_dir = f"datasets/{current_dataset}"
            if not os.path.exists(dataset_dir):
                print(f"Aviso: Diretório {dataset_dir} não encontrado. Pulando dataset {current_dataset}.")
                continue

            all_instances = sorted([f for f in os.listdir(dataset_dir) if f.endswith(".txt")])
            
            if SELECTED_INSTANCES:
                instances = [f for f in all_instances if f in SELECTED_INSTANCES]
            else:
                instances = all_instances
                if MAX_INSTANCES > 0 and len(instances) > MAX_INSTANCES:
                    instances = instances[:MAX_INSTANCES]

            if not instances:
                print(f"Aviso: Nenhuma instância para processar no dataset {current_dataset}.")
                continue
                
            print(f"\n--- Processando Dataset: {current_dataset} ({len(instances)} instâncias) ---")

            for inst_file in instances:
                input_path = os.path.join(dataset_dir, inst_file)
                best_known_key = f"{current_dataset}/{inst_file}"
                best_known = best_values.get(best_known_key, 0.0)
                print(f"Processando {inst_file}:")
                
                for constr in constructives_list:
                    strategy = f"single+{constr}+none"
                    
                    for run_idx in range(1, RUNS_PER_INSTANCE + 1):
                        current_seed = str(int(SEED) + run_idx) if SEED != "0" else "0"
                        temp_output = os.path.join(OUTPUT_DIR, f"temp_{current_dataset}_{inst_file}_{constr}_r{run_idx}.txt")
                        
                        print(f"  [{constr}] Run {run_idx:02d}...", end=" ", flush=True)
                        
                        start_time = time.time()
                        stdout = ""
                        process_status = "OK"
                        score_from_stdout = 0.0
                        
                        try:
                            cmd = [BINARY_PATH, input_path, temp_output, strategy, current_seed]
                            process = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_LIMIT)
                            stdout = process.stdout
                            elapsed = time.time() - start_time
                            
                            if process.returncode != 0:
                                process_status = f"ERROR_{process.returncode}"
                                
                            if "Final Score:" in stdout:
                                score_match = re.search(r"Final Score:\s*([\d\.]+)", stdout)
                                if score_match:
                                    score_from_stdout = float(score_match.group(1))
                            elif "Solution NOT Feasible!" in stdout:
                                process_status = "INFEASIBLE_IN_RUST"
                                
                        except subprocess.TimeoutExpired as e:
                            elapsed = time.time() - start_time
                            process_status = "TIMEOUT"
                            print("TIMEOUT...", end=" ")

                        is_feasible, score, val_status = validate_solution(input_path, temp_output)
                        
                        if process_status == "OK":
                            if val_status == "MISSING_OUTPUT" and score_from_stdout > 0:
                                final_status = "OK_NO_CHECKER"
                                score = score_from_stdout
                            else:
                                final_status = val_status
                                if not is_feasible and val_status != "OK":
                                    score = score_from_stdout
                        else:
                            final_status = process_status
                        
                        gap = ((best_known - score) / best_known * 100.0) if (best_known > 0 and score > 0) else 0.0
                        
                        print(f"Score: {score:.4f} | Gap: {gap:.2f}% | Status: {final_status}")
                        
                        writer.writerow({
                            "dataset": current_dataset,
                            "instance": inst_file, "run": run_idx, "strategy": strategy,
                            "constructive": constr,
                            "score": score, "best_known": best_known, "gap_percent": gap,
                            "time_seconds": elapsed, "status": final_status
                        })
                        csvfile.flush()
                        
                        if os.path.exists(temp_output): 
                            os.remove(temp_output)

    print("-" * 100)
    print(f"Benchmark concluído. Resultados salvos em {output_csv}")

if __name__ == "__main__":
    run_benchmark()
