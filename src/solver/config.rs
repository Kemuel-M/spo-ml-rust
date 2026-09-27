use std::time::Duration;
use crate::solution::ProblemData;
use crate::local_searchs::{SearchStrategy, NeighborhoodType, SearchDimension, LocalSearchConfig};

/// Configurações específicas para GRASP
#[derive(Debug, Clone)]
pub struct GraspConfig {
    pub iterations: usize,
    pub max_time_secs: u64,
    pub chunk_size: usize,
    pub alpha_min: f64,
    pub alpha_max: f64,
    pub reactive_ema: f64,
    pub reactive_power: i32,
    pub reactive_penalty: f64,
}

impl Default for GraspConfig {
    fn default() -> Self {
        Self {
            iterations: 100,
            max_time_secs: 150,
            chunk_size: 20,
            alpha_min: 0.05,
            alpha_max: 0.30,
            reactive_ema: 0.20,
            reactive_power: 6,
            reactive_penalty: 0.8,
        }
    }
}

/// Configurações específicas para ILS
#[derive(Debug, Clone)]
pub struct IlsConfig {
    pub iterations: usize,
    pub max_time_secs: u64,
    pub walkers: usize,
    pub perturbation_base: f64,
    pub perturbation_max: f64,
    pub stagnation_limit: usize,
    pub log_frequency: usize,
}

impl Default for IlsConfig {
    fn default() -> Self {
        Self {
            iterations: 100,
            max_time_secs: 150,
            walkers: 20,
            perturbation_base: 0.15,
            perturbation_max: 0.40,
            stagnation_limit: 15,
            log_frequency: 10,
        }
    }
}

/// Configurações específicas para Simulated Annealing
#[derive(Debug, Clone)]
pub struct SaConfig {
    pub t0: f64,
    pub cooling: f64,
    pub iter_per_temp: usize,
    pub max_time_secs: u64,
    pub min_temp: f64,
    pub survival_fallback_ratio: f64,
    pub stagnation_threshold_divisor: usize,
    pub stagnation_threshold_min: usize,
    pub max_reheats: usize,
    pub initial_acceptance_prob: f64,
    pub reheat_factor_base: f64,
    pub reheat_factor_step: f64,
    pub reheat_factor_max: f64,
    pub kick_strength_divisor: usize,
    pub kick_min_orders: usize,
    pub kick_max_orders: usize,
    pub initial_temp_samples: usize,
    pub log_frequency: usize,
    pub swap_prob_start: f64,
    pub swap_prob_end: f64,
    pub block_prob_start: f64,
    pub block_prob_end: f64,
    pub ins_prob_start: f64,
    pub ins_prob_end: f64,
    pub max_attempts_swap: usize,
    pub max_attempts_ins: usize,
    pub max_attempts_rem: usize,
    pub max_attempts_block: usize,
    pub init_temp_swap_prob: u32,
    pub init_temp_ins_prob: u32,
}

impl Default for SaConfig {
    fn default() -> Self {
        Self {
            t0: 100.0,
            cooling: 0.999,
            iter_per_temp: 100,
            max_time_secs: 150,
            min_temp: 0.0001,
            survival_fallback_ratio: 0.05,
            stagnation_threshold_divisor: 2,
            stagnation_threshold_min: 400,
            max_reheats: 6,
            initial_acceptance_prob: 0.8,
            reheat_factor_base: 0.50,
            reheat_factor_step: 0.1,
            reheat_factor_max: 0.95,
            kick_strength_divisor: 10,
            kick_min_orders: 3,
            kick_max_orders: 10,
            initial_temp_samples: 100,
            log_frequency: 1000,
            swap_prob_start: 40.0,
            swap_prob_end: 70.0,
            block_prob_start: 20.0,
            block_prob_end: 10.0,
            ins_prob_start: 20.0,
            ins_prob_end: 10.0,
            max_attempts_swap: 20,
            max_attempts_ins: 20,
            max_attempts_rem: 15,
            max_attempts_block: 30,
            init_temp_swap_prob: 80,
            init_temp_ins_prob: 10,
        }
    }
}

/// Configurações específicas para Algoritmo Genético
#[derive(Debug, Clone)]
pub struct GaConfig {
    pub pop_size: usize,
    pub generations: usize,
    pub mutation_rate: f64,
    pub max_time_secs: u64,
}

impl Default for GaConfig {
    fn default() -> Self {
        Self {
            pop_size: 50,
            generations: 100,
            mutation_rate: 0.1,
            max_time_secs: 150,
        }
    }
}

/// Configurações específicas para BPSO
#[derive(Debug, Clone)]
pub struct BpsoConfig {
    pub pop_size: usize,
    pub iterations: usize,
    pub w_min: f64,
    pub w_max: f64,
    pub c1: f64,
    pub c2: f64,
    pub c1_escape: f64,
    pub c2_escape: f64,
    pub v_max: f64,
    pub max_time_secs: u64,
    pub stagnation_limit: usize,
    pub stag_threshold_escape: f64,
    pub stag_threshold_panic: f64,
    pub p_orders: f64,
    pub ls_prob: f64,
    pub ls_prob_high: f64,
    pub turbulence_base: f64,
    pub turbulence_high: f64,
    pub log_frequency: usize,
}

impl Default for BpsoConfig {
    fn default() -> Self {
        Self {
            pop_size: 100,
            iterations: 1000,
            w_min: 0.4,
            w_max: 0.9,
            c1: 1.494,
            c2: 1.494,
            c1_escape: 2.0,
            c2_escape: 1.0,
            v_max: 4.0,
            max_time_secs: 600,
            stagnation_limit: 150,
            stag_threshold_escape: 0.3,
            stag_threshold_panic: 0.6,
            p_orders: 0.5,
            ls_prob: 0.15,
            ls_prob_high: 0.3,
            turbulence_base: 0.05,
            turbulence_high: 0.20,
            log_frequency: 10,
        }
    }
}

/// Agregador de configurações de Metaheurísticas
#[derive(Debug, Clone)]
#[derive(Default)]
pub struct MetaConfig {
    pub grasp: GraspConfig,
    pub ils: IlsConfig,
    pub sa: SaConfig,
    pub ga: GaConfig,
    pub bpso: BpsoConfig,
}


/// Centralização de todos os Hiperparâmetros do Solver
#[derive(Debug, Clone)]
pub struct SolverConfig {
    pub max_runtime: Duration,
    pub meta: MetaConfig,
    pub ls: LocalSearchConfig,
}

impl Default for SolverConfig {
    fn default() -> Self {
        let max_runtime = Duration::from_secs(150);
        Self {
            max_runtime,
            meta: MetaConfig::default(),
            ls: LocalSearchConfig {
                dimension: SearchDimension::Orders,
                strategy: SearchStrategy::FirstImprovement,
                neighborhood: NeighborhoodType::Swap,
                sampling_size: 5000,
                max_iterations: 1000,
                max_time_secs: 30,
            },
        }
    }
}

impl SolverConfig {

    pub fn adjust_for_data(&mut self, data: &ProblemData) {
        let n = data.orders.len() as f64;
        
        // Fator de escala: decai hiperbolicamente para n > 1000
        let scale_factor = if n <= 1000.0 { 1.0 } else { 1000.0 / n };

        // Meta-heurísticas Base (Padrão 100, piso de 10)
        let iter_base = (100.0 * scale_factor) as usize;
        self.meta.grasp.iterations = iter_base.max(10);
        self.meta.ils.iterations = iter_base.max(10);
        self.meta.ga.generations = iter_base.max(10);

        // BPSO (Padrão 1000, piso de 50)
        let bpso_iter = (1000.0 * scale_factor) as usize;
        self.meta.bpso.iterations = bpso_iter.max(50);
        
        // Buscas Locais
        let ls_iter = (1000.0 * scale_factor) as usize;
        self.ls.max_iterations = ls_iter.max(50);
        
        let ls_sampling = (5000.0 * scale_factor) as usize;
        self.ls.sampling_size = ls_sampling.max(500);
    }

    pub fn parse_strategy(&mut self, strategy_str: &str) -> (MetaheuristicType, ConstructiveType, LocalSearchType) {
        let parts: Vec<&str> = strategy_str.split('+').collect();
        
        let meta_str = parts.first().unwrap_or(&"single");
        let constr_str = parts.get(1).unwrap_or(&"order_static");
        let ls_str = parts.get(2).unwrap_or(&"none");

        let meta = MetaheuristicType::parse(meta_str, self);
        let constr = std::str::FromStr::from_str(constr_str).unwrap_or(ConstructiveType::OrderStatic);
        let ls = LocalSearchType::parse(ls_str, self);

        (meta, constr, ls)
    }
}

#[derive(Debug, Clone)]
pub enum ConstructiveType {
    OrderStatic,
    OrderAdaptive,
    OrderRandom,
    AisleStatic,
    AisleAdaptive,
    AisleRandom,
    Hybrid,
    SuperHybrid { p_zero_alpha: f64 },
}

#[derive(Debug, Clone)]
pub enum LocalSearchType {
    None,
    HillClimbing { dimension: SearchDimension, strategy: SearchStrategy, neighborhood: NeighborhoodType },
    Vnd { dimension: SearchDimension },
    HVnd,
    TabuSearch { dimension: SearchDimension, tenure: usize, iterations: usize },
    LateAcceptance { dimension: SearchDimension, list_size: usize, iterations: usize },
    HLahc { list_size: usize, iterations: usize },
    HHc,
}

#[derive(Debug, Clone)]
pub enum MetaheuristicType {
    SingleShot, 
    Grasp { iterations: usize },
    Ils { dimension: SearchDimension, iterations: usize },
    SimulatedAnnealing { dimension: SearchDimension, t0: f64, cooling: f64 },
    GeneticAlgorithm { dimension: SearchDimension, pop_size: usize, generations: usize },
    Bpso { dimension: BpsoDimension, pop_size: usize, iterations: usize },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BpsoDimension {
    Orders,
    Aisles,
    Hybrid,
}

impl std::str::FromStr for ConstructiveType {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split(':').collect();
        let name = *parts.first().unwrap_or(&"order_static");
        Ok(match name {
            "order_static" | "os" | "static" => ConstructiveType::OrderStatic,
            "order_adaptive" | "oa" | "adaptive" => ConstructiveType::OrderAdaptive,
            "order_random" | "or" | "random" => ConstructiveType::OrderRandom,
            "aisle_static" | "as" => ConstructiveType::AisleStatic,
            "aisle_adaptive" | "aa" => ConstructiveType::AisleAdaptive,
            "aisle_random" | "ar" => ConstructiveType::AisleRandom,
            "hybrid_random" | "hr" | "hybrid" => ConstructiveType::Hybrid,
            "super_hybrid" | "sh" => {
                let p_zero = parts.get(1).and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.10);
                ConstructiveType::SuperHybrid { p_zero_alpha: p_zero }
            },
            _ => ConstructiveType::OrderStatic,
        })
    }
}

impl MetaheuristicType {
    pub fn parse(s: &str, config: &mut SolverConfig) -> Self {
        let parts: Vec<&str> = s.split(':').collect();
        let m_name = parts[0];
        let m_val1 = parts.get(1).and_then(|v| v.parse::<f64>().ok());
        let m_val2 = parts.get(2).and_then(|v| v.parse::<f64>().ok());
        let m_val3 = parts.get(3).and_then(|v| v.parse::<f64>().ok());

        if let Some(timeout_secs) = m_val3 {
            let dur = Duration::from_secs(timeout_secs as u64);
            config.max_runtime = dur;
            config.meta.grasp.max_time_secs = timeout_secs as u64;
            config.meta.ils.max_time_secs = timeout_secs as u64;
            config.meta.sa.max_time_secs = timeout_secs as u64;
            config.meta.ga.max_time_secs = timeout_secs as u64;
            config.meta.bpso.max_time_secs = timeout_secs as u64;
        }

        let dim = if m_name.starts_with("a_") { SearchDimension::Aisles } else { SearchDimension::Orders };

        match m_name {
            "grasp" | "a_grasp" | "h_grasp" | "hybrid_grasp" => MetaheuristicType::Grasp { 
                iterations: m_val1.map(|v| v as usize).unwrap_or(config.meta.grasp.iterations) 
            },
            "ils" | "a_ils" => MetaheuristicType::Ils { 
                dimension: dim, 
                iterations: m_val1.map(|v| v as usize).unwrap_or(config.meta.ils.iterations) 
            },
            "sa" | "a_sa" => MetaheuristicType::SimulatedAnnealing { 
                dimension: dim, t0: config.meta.sa.t0, cooling: config.meta.sa.cooling 
            },
            "ga" | "a_ga" => MetaheuristicType::GeneticAlgorithm { 
                dimension: dim, 
                pop_size: m_val2.map(|v| v as usize).unwrap_or(config.meta.ga.pop_size), 
                generations: m_val1.map(|v| v as usize).unwrap_or(config.meta.ga.generations) 
            },
            "bpso" | "a_bpso" | "h_bpso" => {
                let b_dim = match m_name {
                    "a_bpso" => BpsoDimension::Aisles,
                    "h_bpso" => BpsoDimension::Hybrid,
                    _ => BpsoDimension::Orders,
                };
                MetaheuristicType::Bpso { 
                    dimension: b_dim, 
                    pop_size: m_val2.map(|v| v as usize).unwrap_or(config.meta.bpso.pop_size), 
                    iterations: m_val1.map(|v| v as usize).unwrap_or(config.meta.bpso.iterations) 
                }
            },
            _ => MetaheuristicType::SingleShot,
        }
    }
}

impl LocalSearchType {
    pub fn parse(s: &str, config: &SolverConfig) -> Self {
        let parts: Vec<&str> = s.split(':').collect();
        let l_name = parts[0];
        let l_val = parts.get(1).and_then(|v| v.parse::<f64>().ok());
        
        let dim = if l_name.starts_with("a_") { SearchDimension::Aisles } else { SearchDimension::Orders };

        match l_name {
            "hvnd" => LocalSearchType::HVnd,
            "hhc" => LocalSearchType::HHc,
            "hlahc" => LocalSearchType::HLahc {
                list_size: 50,
                iterations: l_val.map(|v| v as usize).unwrap_or(1000)
            },
            "vnd" | "a_vnd" => LocalSearchType::Vnd { dimension: dim },
            "tabu" | "a_tabu" => LocalSearchType::TabuSearch { 
                dimension: dim, tenure: 20, 
                iterations: l_val.map(|v| v as usize).unwrap_or(config.ls.max_iterations) 
            },
            "lahc" | "a_lahc" => LocalSearchType::LateAcceptance { 
                dimension: dim, list_size: 50, 
                iterations: l_val.map(|v| v as usize).unwrap_or(config.ls.max_iterations) 
            },
            "swap" | "a_swap" => LocalSearchType::HillClimbing { 
                dimension: dim, strategy: SearchStrategy::BestImprovement, neighborhood: NeighborhoodType::Swap 
            },
            _ => LocalSearchType::None,
        }
    }
}
