import os
import subprocess
import sys
import platform
import csv

# --- CONFIGURAÇÃO ---
MAX_RUNNING_TIME = "600s"
BINARY_NAME = "spo-ml-rust"
# Showcase heuristics for "all" mode
ALL_HEURISTICS = [
    "single+order_static+none", 
    "single+order_adaptive+none", 
    "single+order_static+vnd", 
    "grasp+order_random+vnd",
    "ils+aisle_random+lahc"
]

def compile_code(root_folder):
    print(f"🔨 Compiling Rust code in {root_folder}...")
    result = subprocess.run(
        ["cargo", "build", "--release"],
        capture_output=True,
        text=True,
        cwd=root_folder
    )
    if result.returncode != 0:
        print("❌ Compilation failed:")
        print(result.stderr)
        return False
    print("✅ Compilation successful.\n")
    return True

def run_solver(root_folder, input_path, heuristic_arg=None):
    # Define comando de timeout
    if platform.system() == "Darwin": timeout_command = "gtimeout"
    else: timeout_command = "timeout"

    # Caminho do binário
    exe_ext = ".exe" if platform.system() == "Windows" else ""
    binary_path = os.path.join(root_folder, "target", "release", f"{BINARY_NAME}{exe_ext}")

    if not os.path.exists(binary_path):
        print(f"❌ Binary not found at: {binary_path}")
        return

    # Determina lista de heurísticas a rodar
    heuristics_to_run = []
    if heuristic_arg == "all":
        heuristics_to_run = ALL_HEURISTICS
    elif heuristic_arg:
        heuristics_to_run = [heuristic_arg]
    else:
        heuristics_to_run = ["default"] # Caso seu Rust tenha um default sem argumentos

    # Determina arquivos de entrada
    if os.path.isdir(input_path):
        files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(".txt")]
        files.sort()
        print(f"🚀 Running solver for {len(files)} instances using: {heuristics_to_run}")
    elif os.path.isfile(input_path):
        files = [input_path]
    else:
        print(f"❌ Input path not found: {input_path}")
        return

    # Preparar arquivo de resumo (CSV) para o Artigo
    output_dir_path = os.path.join(root_folder, "outputs")
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    summary_file = os.path.join(output_dir_path, "results_summary.csv")
    
    # Decide se o cabeçalho precisa ser escrito
    write_header = not os.path.exists(summary_file) or os.path.getsize(summary_file) == 0

    # Abre o arquivo no modo de adição ('a') e escreve o cabeçalho apenas se for necessário
    with open(summary_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(['Instance', 'Heuristic', 'Status', 'Objective'])

        # Loop Principal
        for input_file in files:
            print(f"\nProcessing {os.path.basename(input_file)}...")

            for h in heuristics_to_run:
                # Define nome do output (sanitiza + para _)
                safe_h = h.replace('+', '_')
                base_name = os.path.splitext(os.path.basename(input_file))[0]
                output_filename = f"{base_name}_{safe_h}.txt" if h != "default" else f"{base_name}.txt"

                # Cria pasta de output baseada na estrutura de input (datasets/a -> outputs/a)
                output_file = input_file.replace("datasets", "outputs").replace(os.path.basename(input_file), output_filename)
                output_folder = os.path.dirname(output_file)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                cmd = [timeout_command, MAX_RUNNING_TIME, binary_path, input_file, output_file]

                # Adiciona o argumento da heurística se não for default
                if h != "default":
                    cmd.append(h)

                try:
                    # Roda o Rust
                    result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

                    status = "OK"
                    score = 0.0

                    if result.returncode == 124:
                        print(f"   ⏳ [{h}] Timeout")
                        status = "Timeout"
                    elif result.returncode != 0:
                        print(f"   ❌ [{h}] Runtime Error: {result.stderr.strip()}")
                        status = "Error"
                    else:
                        # Se rodou com sucesso, chama o CHECKER imediatamente
                        score = run_checker(input_file, output_file)
                        print(f"   ✅ [{h}] Score: {score:.4f}")

                    # Salva no CSV
                    writer.writerow([base_name, h, status, score])

                except Exception as e:
                    print(f"   ❌ Execution failed: {e}")

    print(f"\n📊 Summary written to {summary_file}")

def run_checker(input_file, output_file):
    try:
        cmd = ["python3", "checker.py", input_file, output_file]
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout.strip()

        if "Is solution feasible: True" in output:
            # Extrai o número após "Objective function value:"
            for line in output.split('\n'):
                if "Objective function value:" in line:
                    return float(line.split(":")[1].strip())
        return 0.0 # Inviável ou erro
    except:
        return 0.0

if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python run_challenge.py <project_root> <input_path> [heuristic]")
        print("       <heuristic> options: static, adaptive, random, aisle, all")
        sys.exit(1)

    root_dir = sys.argv[1]
    input_path_arg = sys.argv[2]
    heuristic_arg = sys.argv[3] if len(sys.argv) == 4 else None

    if compile_code(root_dir):
        run_solver(root_dir, input_path_arg, heuristic_arg)
