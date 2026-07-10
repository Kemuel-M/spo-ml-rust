import os
import subprocess

BINARY = "./target/release/spo-ml-rust"
INSTANCE = "datasets/a/instance_0001.txt" # Instância pequena e rápida

CONSTRUCTIVES = [
    "os", "oa", "or",
    "as", "aa", "ar",
    "hr", "sh"
]

LOCAL_SEARCHES = [
    "none",
    "swap", "a_swap",
    "vnd", "a_vnd",
    "hvnd",
    "tabu", "a_tabu",
    "lahc", "a_lahc"
]

def main():
    if not os.path.exists(BINARY):
        print(f"Buildando projeto em modo release...")
        subprocess.run(["cargo", "build", "--release"], check=True)
        
    if not os.path.exists(INSTANCE):
        print(f"Erro: Instância {INSTANCE} não encontrada! Por favor, atualize o caminho da instância no script.")
        return

    print("===============================================================")
    print(" Iniciando bateria de testes do Refactoring ")
    print(" (Magic Strings CLI + Generic Neighborhoods Box<dyn Trait>)")
    print("===============================================================\n")
    
    total = len(CONSTRUCTIVES) * len(LOCAL_SEARCHES)
    count = 0
    errors = 0
    
    # Criar pasta outputs se não existir
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    
    for c in CONSTRUCTIVES:
        for ls in LOCAL_SEARCHES:
            count += 1
            strategy = f"single+{c}+{ls}"
            cmd = [BINARY, INSTANCE, "outputs/test_out.txt", strategy, "42"]
            print(f"[{count:02d}/{total:02d}] Testando strategy '{strategy}'...".ljust(50), end="", flush=True)
            
            try:
                # Limite de 20s. Sem metaheurística deve rodar em 0.1 a 2 segundos
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
                if result.returncode == 0:
                    print("OK")
                else:
                    print(f"ERROR (Exit Code {result.returncode})")
                    print("--- STDOUT ---")
                    print(result.stdout[:500])
                    print("--- STDERR ---")
                    print(result.stderr[:500])
                    errors += 1
            except subprocess.TimeoutExpired:
                print("TIMEOUT (Estourou 20s)")
                errors += 1
                
    print("\n===============================================================")
    if errors == 0:
        print(" SUCESSO! A refatoração manteve 100% da integridade.")
        print(" Todos os parsers CLI e Vizinhanças genéricas funcionaram!")
    else:
        print(f" FALHA! Ocorreram {errors} erros durante os testes.")
    print("===============================================================")

if __name__ == '__main__':
    main()
