import os
import subprocess
import time
import csv
import re

# =============================================================================
# CONFIGURAÇÕES EDITÁVEIS
# =============================================================================

# --- Dataset e Execução ---
DATASET = "a"               # Opções: "a", "b", "x"

# Lista de instâncias específicas para rodar. Se a lista não estiver vazia, rodará apenas estas instâncias.
# Deixe vazia (SELECTED_INSTANCES = []) para rodar todas as instâncias do dataset (sujeito ao MAX_INSTANCES).
# Exemplo: SELECTED_INSTANCES = ["instance_0014.txt", "instance_0015.txt"]
#SELECTED_INSTANCES = ["instance_0001.txt", "instance_0002.txt", "instance_0003.txt", "instance_0004.txt", "instance_0005.txt", "instance_0006.txt", "instance_0007.txt", "instance_0008.txt", "instance_0009.txt", "instance_0010.txt"]
#SELECTED_INSTANCES = ["instance_0011.txt", "instance_0012.txt", "instance_0013.txt", "instance_0014.txt", "instance_0015.txt"]
SELECTED_INSTANCES = ["instance_0010.txt"]

SEED = "0"                  # Semente aleatória (use "0" para semente do sistema)
TIMEOUT_LIMIT = 600         # Tempo máximo por instância em segundos (PASSADO PARA O RUST)

# Limite de instâncias a processar (0 para processar todas do diretório).
# Nota: Este parâmetro é ignorado caso SELECTED_INSTANCES não esteja vazia.
MAX_INSTANCES = 0           

RUNS_PER_INSTANCE = 3      # Quantidade de vezes que cada instância será executada

# --- Parâmetros da Meta-heurística ---
# Meta-heurística base
META_BASE = "h_bpso"        # Ex: "h_bpso", "bpso", "a_bpso"

ITERATIONS = 1000           # Número de iterações do BPSO
POP_SIZE = 100              # Tamanho da população

# Estratégias base para população inicial (Construtivos)
# Opções implementadas (nome extenso ou sigla):
# - "order_static"   ou "os"
# - "order_adaptive" ou "oa"
# - "order_random"   ou "or"
# - "aisle_static"   ou "as"
# - "aisle_adaptive" ou "aa"
# - "aisle_random"   ou "ar"
# - "hybrid_random"  ou "hr"
# - "super_hybrid"   ou "sh" (agora aceita parâmetro opcional de chance de alpha zero, ex: "sh:0.05")
# Pode ser uma string única ou uma lista de strings. O script testará todas as combinações.
CONSTRUCTIVES = ["sh:0.05"]

# Estratégias de busca local / memético
# Pode ser uma string única ou uma lista de strings. O script testará todas as combinações.
LOCAL_SEARCHES = ["hvnd"]   # ["none", "vnd", "hvnd", "tabu", "lahc", "swap", "insert", "remove"]

# --- Caminhos de Arquivos ---
BINARY_PATH = "./target/release/spo-ml-rust"
BEST_VAL_PATH = "best_solutions/best_objectives.csv"
CHECKER_PATH = "runs/checker.py"
OUTPUT_DIR = "outputs"

# =============================================================================
# DERIVADOS
# =============================================================================

DATASETS_DIR = f"datasets/{DATASET}"


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
    if not os.path.exists(OUTPUT_DIR): 
        os.makedirs(OUTPUT_DIR)
        
    best_values = load_best_values()
    
    if not os.path.exists(DATASETS_DIR):
        print(f"Erro: Diretorio {DATASETS_DIR} nao encontrado.")
        return

    all_instances = sorted([f for f in os.listdir(DATASETS_DIR) if f.endswith(".txt")])
    
    if SELECTED_INSTANCES:
        instances = [f for f in all_instances if f in SELECTED_INSTANCES]
        # Validar se as instâncias selecionadas realmente existem no diretório
        for inst in SELECTED_INSTANCES:
            if inst not in all_instances:
                print(f"Aviso: Instância {inst} não encontrada no diretório {DATASETS_DIR}")
    else:
        instances = all_instances
        if MAX_INSTANCES > 0 and len(instances) > MAX_INSTANCES:
            instances = instances[:MAX_INSTANCES]

    if not instances:
        print("Nenhuma instância para processar.")
        return
        
    # Garante que sejam listas mesmo se o usuário configurar apenas uma string
    constructives_list = CONSTRUCTIVES if isinstance(CONSTRUCTIVES, list) else [CONSTRUCTIVES]
    ls_list = LOCAL_SEARCHES if isinstance(LOCAL_SEARCHES, list) else [LOCAL_SEARCHES]

    print(f"Iniciando Benchmark [{DATASET.upper()}]")
    print(f"Meta-heurística Base: {META_BASE}")
    print(f"Construtivos: {', '.join(constructives_list)}")
    print(f"Buscas Locais: {', '.join(ls_list)}")
    if SELECTED_INSTANCES:
        print(f"Instâncias ({len(instances)} selecionadas): {', '.join(instances)}")
    else:
        print(f"Instâncias: {len(instances)}")
    print(f"Rodadas por Instância: {RUNS_PER_INSTANCE}")
    print("-" * 100)

    fieldnames = [
        "instance", "run", "strategy", "meta", "constructive", "local_search", 
        "score", "total_iterations", 
        "final_avg_obj", "final_avg_v", "ls_hit_rate_final", "imp_rate_final",
        "final_v_sat", "final_h_dist",
        "orders_split", "aisles_split",
        "best_known", "gap_percent", "time_seconds", "status"
    ]
    
    # Criar um nome detalhado com todas as configurações utilizadas
    constructives_str = "-".join(constructives_list)
    ls_str = "-".join(ls_list)
    suffix = "_selected" if SELECTED_INSTANCES else ""
    
    base_csv_name = f"bench_{META_BASE}_{DATASET}_it{ITERATIONS}_p{POP_SIZE}_t{TIMEOUT_LIMIT}_c[{constructives_str}]_ls[{ls_str}]{suffix}"
    
    # Lógica para não sobrescrever caso o arquivo já exista
    counter = 1
    output_csv = f"{OUTPUT_DIR}/{base_csv_name}.csv"
    while os.path.exists(output_csv):
        output_csv = f"{OUTPUT_DIR}/{base_csv_name}_{counter}.csv"
        counter += 1

    with open(output_csv, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for inst_file in instances:
            input_path = os.path.join(DATASETS_DIR, inst_file)
            best_known = best_values.get(inst_file, 0.0)
            print(f"Processando {inst_file}:")
            
            for constr in constructives_list:
                for ls in ls_list:
                    # Formato: META:ITERS:POP:TIME + CONSTR + LS
                    strategy = f"{META_BASE}:{ITERATIONS}:{POP_SIZE}:{TIMEOUT_LIMIT}+{constr}+{ls}"
                    
                    for run_idx in range(1, RUNS_PER_INSTANCE + 1):
                        current_seed = str(int(SEED) + run_idx) if SEED != "0" else "0"
                        temp_output = os.path.join(OUTPUT_DIR, f"temp_{inst_file}_{constr}_{ls}_r{run_idx}.txt")
                        
                        print(f"  [{constr}+{ls}] Run {run_idx:02d}...", end=" ", flush=True)
                        
                        start_time = time.time()
                        stdout = ""
                        process_status = "OK"
                        
                        try:
                            # Damos 60 segundos de folga para o Python, pois o Rust deve parar sozinho antes
                            cmd = [BINARY_PATH, input_path, temp_output, strategy, current_seed]
                            process = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_LIMIT + 60)
                            stdout = process.stdout
                            elapsed = time.time() - start_time
                            
                            if process.returncode != 0:
                                process_status = f"ERROR_{process.returncode}"
                        except subprocess.TimeoutExpired as e:
                            # Garante que stdout seja string mesmo em caso de bytes no timeout
                            raw_stdout = e.stdout if e.stdout else b""
                            stdout = raw_stdout.decode('utf-8', errors='ignore') if isinstance(raw_stdout, bytes) else raw_stdout
                            elapsed = time.time() - start_time
                            process_status = "TIMEOUT"
                            print("TIMEOUT (Python killed)...", end=" ")

                        # Extrair métricas do stdout
                        iter_match = re.search(r"\[BPSO\] Finished in (\d+) iterations", stdout)
                        total_iters = iter_match.group(1) if iter_match else ""
                        
                        hybrid_match = re.search(r"\[Hybrid Mode: (\d+) Orders / (\d+) Aisles\]", stdout)
                        o_split = hybrid_match.group(1) if hybrid_match else ""
                        a_split = hybrid_match.group(2) if hybrid_match else ""

                        log_lines = re.findall(r"Iter\s+\d+\s+\|\s+Best:\s+[\d\.]+\s+\|\s+Imp:\s+([\d\.]+)%\s+\|\s+Stag:\s+\d+\s+\|\s+LSHit:\s+([\d\.]+)%", stdout, re.MULTILINE)
                        final_avg_obj, final_imp_rate, final_avg_v, final_ls_hit = [""] * 4
                        if log_lines:
                            final_imp_rate, final_ls_hit = log_lines[-1]

                        metrics_match = re.search(r"\[BPSO Metrics\] Final Velocity Saturation: ([\d\.]+)% \| Final Diversity \(HDist\): ([\d\.]+)%", stdout)
                        final_v_sat = metrics_match.group(1) if metrics_match else ""
                        final_h_dist = metrics_match.group(2) if metrics_match else ""

                        # Validar solução
                        is_feasible, score, val_status = validate_solution(input_path, temp_output)
                        
                        final_status = val_status if process_status == "OK" else process_status
                        
                        gap = ((best_known - score) / best_known * 100.0) if best_known > 0 else 0.0
                        
                        print(f"Score: {score:.4f} | Gap: {gap:.2f}% | Status: {final_status}")
                        
                        writer.writerow({
                            "instance": inst_file, "run": run_idx, "strategy": strategy,
                            "meta": META_BASE, "constructive": constr, "local_search": ls,
                            "score": score, "total_iterations": total_iters,
                            "final_avg_obj": final_avg_obj, "final_avg_v": final_avg_v,
                            "ls_hit_rate_final": final_ls_hit, "imp_rate_final": final_imp_rate,
                            "final_v_sat": final_v_sat, "final_h_dist": final_h_dist,
                            "orders_split": o_split, "aisles_split": a_split,
                            "best_known": best_known, "gap_percent": gap,
                            "time_seconds": elapsed, "status": final_status
                        })
                        csvfile.flush()
                        if os.path.exists(temp_output): os.remove(temp_output)

    print("-" * 100)
    print(f"Benchmark concluído. Resultados salvos em {output_csv}")

if __name__ == "__main__":
    run_benchmark()
