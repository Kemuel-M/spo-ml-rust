use crate::solution::{ChallengeSolution, ProblemData};
use crate::heuristics::LocalSearchAlgorithm;

pub mod tabu_search;
pub mod late_acceptance;
pub mod vnd;
pub mod hybrid_vnd;
pub mod configurable;

pub use configurable::{ConfigurableLocalSearch, generate_neighborhood};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchDimension {
    Orders,
    Aisles,
}

#[derive(Clone, Copy, Debug)]
pub enum SearchStrategy {
    FirstImprovement,
    BestImprovement,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NeighborhoodType {
    Swap,
    Insertion,
    Removal,
}


/// Configuração modular para buscas locais
#[derive(Clone, Debug)]
pub struct LocalSearchConfig {
    pub dimension: SearchDimension,
    pub strategy: SearchStrategy,
    pub neighborhood: NeighborhoodType,
    pub sampling_size: usize,
    pub max_iterations: usize,
    pub max_time_secs: u64,
}

pub struct NoOpLocalSearch;
impl LocalSearchAlgorithm for NoOpLocalSearch {
    fn name(&self) -> String { "None".to_string() }
    fn refine(&self, _solution: &mut ChallengeSolution, _data: &ProblemData, _seed: u64) -> bool { false }
}
