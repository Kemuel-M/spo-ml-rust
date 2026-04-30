use crate::solution::{ChallengeSolution, ProblemData};
use crate::heuristics::{SolverStrategy, ConstructiveAlgorithm, LocalSearchAlgorithm};
use crate::local_searchs::{ConfigurableLocalSearch, NeighborhoodType, LocalSearchConfig, SearchDimension};
use crate::evaluator::Evaluator;

pub mod config;
pub mod simple_solver;

pub use config::{SolverConfig, MetaheuristicType, ConstructiveType, LocalSearchType, GraspConfig};
pub use simple_solver::SimpleSolver;

// --- Challenge Solver ---

pub struct ChallengeSolver<'a> {
    data: &'a ProblemData,
    config: SolverConfig,
}

impl<'a> ChallengeSolver<'a> {
    pub fn new(data: &'a ProblemData) -> Self {
        let mut config = SolverConfig::default();
        config.adjust_for_data(data);
        Self {
            data,
            config,
        }
    }

    pub fn solve(&self, 
        meta: MetaheuristicType, 
        constr: ConstructiveType, 
        ls: LocalSearchType,
        seed: u64
    ) -> ChallengeSolution {
        let solver_instance = self.build(meta, constr, ls);
        solver_instance.solve(self.data, seed)
    }

    fn build(&self, 
        meta_type: MetaheuristicType, 
        constr_type: ConstructiveType, 
        ls_type: LocalSearchType
    ) -> Box<dyn SolverStrategy> {

        use crate::heuristics::timer::{TimedConstructive, TimedLocalSearch, TimedSolver};

        let is_metaheuristic = match meta_type {
            MetaheuristicType::SingleShot => false,
            _ => true,
        };

        // Prepara configuração de Busca Local
        let mut current_ls_config = self.config.ls.clone();
        if is_metaheuristic {
            current_ls_config.max_iterations = (current_ls_config.max_iterations / 10).max(50);
            current_ls_config.sampling_size = (current_ls_config.sampling_size / 2).max(1000);
        }

        let raw_constructive: Box<dyn ConstructiveAlgorithm> = match constr_type {
            ConstructiveType::Static => Box::new(crate::heuristics::static_greedy::StaticGreedy),
            ConstructiveType::Adaptive => Box::new(crate::heuristics::adaptive_greedy::AdaptiveGreedy),
            ConstructiveType::Random => Box::new(crate::heuristics::random_greedy::RandomGreedy { alpha: 0.1 }), // Alpha default, será sobrescrito pelo GRASP
            ConstructiveType::AisleStatic => Box::new(crate::heuristics::aisle_centric_static::AisleCentricStatic),
            ConstructiveType::AisleAdaptive => Box::new(crate::heuristics::aisle_centric_adaptive::AisleCentricAdaptive),
            ConstructiveType::AisleRandom => Box::new(crate::heuristics::aisle_centric_random::AisleCentricRandom { alpha: 0.1 }),
            ConstructiveType::Hybrid => Box::new(crate::heuristics::hybrid_random::HybridRandom::new(0.1)),
        };

        let raw_local_search: Box<dyn LocalSearchAlgorithm> = match ls_type {
            LocalSearchType::None => Box::new(crate::local_searchs::NoOpLocalSearch),
            LocalSearchType::HillClimbing { dimension, strategy, neighborhood } => {
                current_ls_config.dimension = dimension;
                current_ls_config.strategy = strategy;
                current_ls_config.neighborhood = neighborhood;
                Box::new(ConfigurableLocalSearch { config: current_ls_config })
            },
            LocalSearchType::Vnd { dimension } => {
                let mut v_config = current_ls_config.clone();
                v_config.dimension = dimension;
                Box::new(crate::local_searchs::vnd::VariableNeighborhoodDescent {
                    neighborhoods: vec![
                        ConfigurableLocalSearch { config: LocalSearchConfig { neighborhood: NeighborhoodType::Insertion, ..v_config.clone() } },
                        ConfigurableLocalSearch { config: LocalSearchConfig { neighborhood: NeighborhoodType::Swap, ..v_config.clone() } },
                        ConfigurableLocalSearch { config: LocalSearchConfig { neighborhood: NeighborhoodType::Removal, ..v_config.clone() } },
                    ],
                    max_time_secs: v_config.max_time_secs,
                })
            },
            LocalSearchType::HVnd => {
                Box::new(crate::local_searchs::hybrid_vnd::HybridVND {
                    neighborhoods: vec![
                        ConfigurableLocalSearch { config: LocalSearchConfig { dimension: SearchDimension::Orders, neighborhood: NeighborhoodType::Insertion, ..current_ls_config.clone() } },
                        ConfigurableLocalSearch { config: LocalSearchConfig { dimension: SearchDimension::Orders, neighborhood: NeighborhoodType::Swap, ..current_ls_config.clone() } },
                        ConfigurableLocalSearch { config: LocalSearchConfig { dimension: SearchDimension::Aisles, neighborhood: NeighborhoodType::Removal, ..current_ls_config.clone() } },
                        ConfigurableLocalSearch { config: LocalSearchConfig { dimension: SearchDimension::Aisles, neighborhood: NeighborhoodType::Swap, ..current_ls_config.clone() } },
                    ],
                    max_time_secs: current_ls_config.max_time_secs,
                })
            },
            LocalSearchType::TabuSearch { dimension, tenure, iterations } => {
                let mut t_config = current_ls_config.clone();
                t_config.dimension = dimension;
                t_config.max_iterations = iterations;
                Box::new(crate::local_searchs::tabu_search::TabuSearch { config: t_config, tenure })
            },
            LocalSearchType::LateAcceptance { dimension, list_size, iterations } => {
                let mut l_config = current_ls_config.clone();
                l_config.dimension = dimension;
                l_config.max_iterations = iterations;
                Box::new(crate::local_searchs::late_acceptance::LateAcceptanceHillClimbing { config: l_config, list_size })
            },
        };

        // Só adiciona cronômetros se NÃO for meta-heurística (para evitar poluir o log no GRASP/GA)
        let (constructive, local_search) = if is_metaheuristic {
            (raw_constructive, raw_local_search)
        } else {
            (Box::new(TimedConstructive::new(raw_constructive)) as Box<dyn ConstructiveAlgorithm>, 
             Box::new(TimedLocalSearch::new(raw_local_search)) as Box<dyn LocalSearchAlgorithm>)
        };

        let strategy: Box<dyn SolverStrategy> = match meta_type {
            MetaheuristicType::SingleShot => Box::new(SimpleSolver { constructive, local_search }),
            MetaheuristicType::Grasp { iterations } => {
                let mut grasp_config = self.config.meta.grasp.clone();
                grasp_config.iterations = iterations;
                Box::new(crate::metaheuristics::Grasp { 
                    constructive, local_search, config: grasp_config
                })
            },
            MetaheuristicType::Ils { dimension, iterations } => Box::new(crate::metaheuristics::Ils { 
                constructive, local_search, dimension, max_iterations: iterations, max_time_secs: self.config.meta.ils.max_time_secs 
            }),
            MetaheuristicType::SimulatedAnnealing { dimension, t0, cooling } => {
                let mut sa_config = self.config.meta.sa.clone();
                sa_config.t0 = t0; sa_config.cooling = cooling;
                Box::new(crate::metaheuristics::simulated_annealing::SimulatedAnnealing { constructive, local_search, dimension, config: sa_config })
            },
            MetaheuristicType::GeneticAlgorithm { dimension, pop_size, generations } => Box::new(crate::metaheuristics::genetic::GeneticAlgorithm {
                constructive, local_search, config: crate::metaheuristics::genetic::GAConfig {
                    dimension, population_size: pop_size, generations: generations,
                    mutation_rate: self.config.meta.ga.mutation_rate, elitism_count: (pop_size / 10).max(1),
                    max_time_secs: self.config.meta.ga.max_time_secs, memetic_prob: 0.2, repair_search_limit: 1000, seed: 42, 
                },
            }),
            MetaheuristicType::Bpso { dimension, pop_size, iterations } => Box::new(crate::metaheuristics::bpso::BPSO {
                constructive, local_search, config: crate::metaheuristics::bpso::BPSOConfig {
                    dimension, population_size: pop_size, iterations: iterations,
                    w_min: self.config.meta.bpso.w_min, w_max: self.config.meta.bpso.w_max,
                    c1: self.config.meta.bpso.c1, c2: self.config.meta.bpso.c2,
                    v_max: self.config.meta.bpso.v_max, ls_prob: 0.15, max_time_secs: self.config.meta.bpso.max_time_secs,
                },
            })
        };

        Box::new(TimedSolver::new(strategy))
    }

    pub fn is_solution_feasible(&self, solution: &ChallengeSolution) -> bool {
        let evaluator = crate::evaluator::StockBalanceEvaluator::new(solution, self.data);
        evaluator.is_stock_valid() && evaluator.current_total_items() >= self.data.wave_size_lb && evaluator.current_total_items() <= self.data.wave_size_ub
    }

    pub fn compute_objective_function(&self, solution: &ChallengeSolution) -> f64 {
        let evaluator = crate::evaluator::StockBalanceEvaluator::new(solution, self.data);
        evaluator.current_objective()
    }
}
