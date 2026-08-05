use crate::solution::{ChallengeSolution, ProblemData};
use crate::heuristics::LocalSearchAlgorithm;
use crate::local_searchs::{ConfigurableLocalSearch, SearchStrategy, NeighborhoodType, LocalSearchConfig, SearchDimension};
use crate::evaluator::{StockBalanceEvaluator, Evaluator, Move};
use std::time::Instant;
use log::{debug, trace};

pub struct HybridVND {
    pub neighborhoods: Vec<ConfigurableLocalSearch>,
    pub max_time_secs: u64,
}

impl HybridVND {
    /// Cria um VND híbrido que alterna entre vizinhanças de Pedidos e Corredores
    pub fn default() -> Self {
        let mut nbh = Vec::new();
        
        let orders_config = LocalSearchConfig {
            dimension: SearchDimension::Orders,
            strategy: SearchStrategy::FirstImprovement,
            neighborhood: NeighborhoodType::Swap,
            sampling_size: 5000,
            max_iterations: 100,
            max_time_secs: 150,
        };

        let aisles_config = LocalSearchConfig {
            dimension: SearchDimension::Aisles,
            strategy: SearchStrategy::FirstImprovement,
            neighborhood: NeighborhoodType::Swap,
            sampling_size: 5000,
            max_iterations: 100,
            max_time_secs: 150,
        };

        // Ordem sugerida: Inserção Pedidos -> Troca Pedidos -> Remoção Corredores -> Troca Corredores
        nbh.push(ConfigurableLocalSearch { 
            config: LocalSearchConfig { neighborhood: NeighborhoodType::Insertion, ..orders_config.clone() },
            neighborhood_engine: crate::local_searchs::neighborhood::build_neighborhood(SearchDimension::Orders, NeighborhoodType::Insertion),
        });
        nbh.push(ConfigurableLocalSearch { 
            config: LocalSearchConfig { neighborhood: NeighborhoodType::Swap, ..orders_config.clone() },
            neighborhood_engine: crate::local_searchs::neighborhood::build_neighborhood(SearchDimension::Orders, NeighborhoodType::Swap),
        });
        nbh.push(ConfigurableLocalSearch { 
            config: LocalSearchConfig { neighborhood: NeighborhoodType::Removal, ..aisles_config.clone() },
            neighborhood_engine: crate::local_searchs::neighborhood::build_neighborhood(SearchDimension::Aisles, NeighborhoodType::Removal),
        });
        nbh.push(ConfigurableLocalSearch { 
            config: LocalSearchConfig { neighborhood: NeighborhoodType::Swap, ..aisles_config.clone() },
            neighborhood_engine: crate::local_searchs::neighborhood::build_neighborhood(SearchDimension::Aisles, NeighborhoodType::Swap),
        });

        Self { neighborhoods: nbh, max_time_secs: 150 }
    }
}

impl LocalSearchAlgorithm for HybridVND {
    fn name(&self) -> String { "Hybrid VND".to_string() }

    fn refine(&self, solution: &mut ChallengeSolution, data: &ProblemData, seed: u64) -> bool {
        let start_time = Instant::now();
        let mut eval = StockBalanceEvaluator::new(solution, data);
        debug!("HVND Iniciado: score base={:.4}", eval.current_objective());
        
        let mut k = 0;
        let mut global_improvement = false;

        while k < self.neighborhoods.len() {
            if start_time.elapsed().as_secs() >= self.max_time_secs {
                break;
            }

            let search_method = &self.neighborhoods[k];
            let improved = search_method.refine(solution, data, seed + k as u64);

            if improved {
                global_improvement = true;
                debug!("HVND: melhoria encontrada na vizinhança {}, novo score={:.4}", k, eval.current_objective());
                k = 0; // Volta para o início se melhorou
            } else {
                k += 1;
            }
        }

        global_improvement
    }
}
