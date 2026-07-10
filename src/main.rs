use anyhow::Result;
use clap::Parser;
use spo_ml_rust::solver::ChallengeSolver;
use spo_ml_rust::io;

/// Solver for the Wave Order Picking Problem
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the input file
    input: String,

    /// Path to the output file
    output: String,

    /// Strategy configuration in format META+CONSTRUCTIVE+LOCALSEARCH
    /// Ex: grasp+order_random+vnd
    #[arg(default_value = "single+order_static+none")]
    strategy_config: String,

    /// Random seed (0 for system time)
    #[arg(default_value_t = 42)]
    seed: u64,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let input_path = &args.input;
    let output_path = &args.output;
    let strategy_str = &args.strategy_config;
    let mut seed = args.seed;

    if seed == 0 {
        use rand::RngCore;
        seed = rand::rng().next_u64();
        println!("Using random seed: {}", seed);
    }

    let is_deterministic = args.seed != 0;
    let data = io::read_input(input_path, is_deterministic)?;
    
    // Instancia configurações padrão
    use spo_ml_rust::solver::SolverConfig;
    let mut config = SolverConfig::default();
    config.adjust_for_data(&data); // Ajustes automáticos por tamanho

    let (meta, constr, ls) = config.parse_strategy(strategy_str);

    let solver = ChallengeSolver::new(&data, config);
    
    println!("--- Wave Order Picking Solver ---");
    println!("Instance: {}", input_path);
    println!("Strategy: {}", strategy_str);
    println!("Seed: {}", seed);
    println!("---------------------------------");

    let solution = solver.solve(meta, constr, ls, seed);

    if solver.is_solution_feasible(&solution) {
        let obj = solver.compute_objective_function(&solution);
        println!("Final Score: {:.4}", obj);
    } else {
        println!("Solution NOT Feasible!");
    }

    io::write_output(&solution, output_path)?;

    Ok(())
}


