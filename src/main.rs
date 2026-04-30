use anyhow::Result;
use std::env;
use spo_ml_rust::solver::{ChallengeSolver, MetaheuristicType, ConstructiveType, LocalSearchType};
use spo_ml_rust::local_searchs::{SearchStrategy, NeighborhoodType};
use spo_ml_rust::io;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 || args.len() > 5 {
        print_usage(&args[0]);
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];
    
    let strategy_str = args.get(3).map_or("single+static+none", |s| s.as_str());
    let mut seed: u64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(42);

    if seed == 0 {
        use rand::RngCore;
        seed = rand::rng().next_u64();
        println!("Using random seed: {}", seed);
    }

    let data = io::read_input(input_path)?;
    
    // Instancia configurações padrão
    use spo_ml_rust::solver::SolverConfig;
    let mut config = SolverConfig::default();
    config.adjust_for_data(&data); // Ajustes automáticos por tamanho

    let (meta, constr, ls) = parse_strategy(strategy_str, &config);

    let solver = ChallengeSolver::new(&data);
    
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

fn print_usage(prog_name: &str) {
    eprintln!("Usage: {} <input> <output> [strategy_config] [seed]", prog_name);
    eprintln!("\nStrategy Configuration Format: META+CONSTRUCTIVE+LOCALSEARCH");
    eprintln!("\nExamples:");
    eprintln!("  grasp+random+vnd        (GRASP with Random Greedy and VND)");
    eprintln!("  single+adaptive+tabu    (One shot Adaptive Greedy refined by Tabu Search)");
    eprintln!("\nAvailable Components:");
    eprintln!("  META: single, grasp, ils, sa, ga, bpso");
    eprintln!("  CONSTRUCTIVE: static, adaptive, random, aisle, aisle_adaptive, aisle_random");
    eprintln!("  LOCALSEARCH: none, hvnd, vnd, tabu, lahc, swap, insert, remove");
}

fn parse_strategy(config_str: &str, defaults: &spo_ml_rust::solver::SolverConfig) -> (MetaheuristicType, ConstructiveType, LocalSearchType) {
    let parts: Vec<&str> = config_str.split('+').collect();
    
    use spo_ml_rust::local_searchs::SearchDimension;

    // Processamento do MetaheuristicType
    let meta_full = parts.get(0).unwrap_or(&"single");
    let m_parts: Vec<&str> = meta_full.split(':').collect();
    let m_name = m_parts[0];
    let m_val = m_parts.get(1).and_then(|v| v.parse::<f64>().ok());

    let meta = match m_name {
        "grasp" | "a_grasp" => MetaheuristicType::Grasp { 
            iterations: m_val.map(|v| v as usize).unwrap_or(defaults.meta.grasp.iterations) 
        },
        "ils" | "a_ils" => {
            let dim = if m_name.starts_with('a') { SearchDimension::Aisles } else { SearchDimension::Orders };
            MetaheuristicType::Ils { 
                dimension: dim, 
                iterations: m_val.map(|v| v as usize).unwrap_or(defaults.meta.ils.iterations) 
            }
        },
        "sa" | "a_sa" => {
            let dim = if m_name.starts_with('a') { SearchDimension::Aisles } else { SearchDimension::Orders };
            MetaheuristicType::SimulatedAnnealing { dimension: dim, t0: defaults.meta.sa.t0, cooling: defaults.meta.sa.cooling }
        },
        "ga" | "a_ga" => {
            let dim = if m_name.starts_with('a') { SearchDimension::Aisles } else { SearchDimension::Orders };
            MetaheuristicType::GeneticAlgorithm { 
                dimension: dim, 
                pop_size: defaults.meta.ga.pop_size, 
                generations: m_val.map(|v| v as usize).unwrap_or(defaults.meta.ga.generations) 
            }
        },
        "bpso" | "a_bpso" => {
            let dim = if m_name.starts_with('a') { SearchDimension::Aisles } else { SearchDimension::Orders };
            MetaheuristicType::Bpso { 
                dimension: dim, 
                pop_size: defaults.meta.bpso.pop_size, 
                iterations: m_val.map(|v| v as usize).unwrap_or(defaults.meta.bpso.iterations) 
            }
        },
        _ => MetaheuristicType::SingleShot,
    };

    // Processamento do ConstructiveType
    let constr_full = parts.get(1).unwrap_or(&"static");
    let c_name = constr_full.split(':').next().unwrap_or("static");

    let constr = match c_name {
        "static" => ConstructiveType::Static,
        "adaptive" => ConstructiveType::Adaptive,
        "random" => ConstructiveType::Random,
        "aisle" | "aisle_static" => ConstructiveType::AisleStatic,
        "aisle_adaptive" => ConstructiveType::AisleAdaptive,
        "aisle_random" => ConstructiveType::AisleRandom,
        "hybrid" | "hybrid_random" => ConstructiveType::Hybrid,
        _ => ConstructiveType::Static,
    };

    // Processamento do LocalSearchType
    let ls_full = parts.get(2).unwrap_or(&"none");
    let l_parts: Vec<&str> = ls_full.split(':').collect();
    let l_name = l_parts[0];
    let l_val = l_parts.get(1).and_then(|v| v.parse::<f64>().ok());

    let ls = match l_name {
        "hvnd" => LocalSearchType::HVnd,
        "vnd" | "a_vnd" => {
            let dim = if l_name.starts_with('a') { SearchDimension::Aisles } else { SearchDimension::Orders };
            LocalSearchType::Vnd { dimension: dim }
        },
        "tabu" | "a_tabu" => {
            let dim = if l_name.starts_with('a') { SearchDimension::Aisles } else { SearchDimension::Orders };
            LocalSearchType::TabuSearch { 
                dimension: dim, 
                tenure: 20, // Tabu tenure fixo ou vir do config? Vamos usar fixo por agora ou defaults.meta?
                iterations: l_val.map(|v| v as usize).unwrap_or(defaults.ls.max_iterations) 
            }
        },
        "lahc" | "a_lahc" => {
            let dim = if l_name.starts_with('a') { SearchDimension::Aisles } else { SearchDimension::Orders };
            LocalSearchType::LateAcceptance { 
                dimension: dim, 
                list_size: 50, 
                iterations: l_val.map(|v| v as usize).unwrap_or(defaults.ls.max_iterations) 
            }
        },
        "swap" | "a_swap" => {
            let dim = if l_name.starts_with('a') { SearchDimension::Aisles } else { SearchDimension::Orders };
            LocalSearchType::HillClimbing { dimension: dim, strategy: SearchStrategy::BestImprovement, neighborhood: NeighborhoodType::Swap }
        },
        _ => LocalSearchType::None,
    };

    (meta, constr, ls)
}
