use crate::solution::{ChallengeSolution, ProblemData};
use crate::heuristics::LocalSearchAlgorithm;
use crate::local_searchs::{ConfigurableLocalSearch, SearchStrategy, NeighborhoodType, LocalSearchConfig, SearchDimension};
use std::time::Instant;

pub struct VariableNeighborhoodDescent {
    pub neighborhoods: Vec<ConfigurableLocalSearch>,
    pub max_time_secs: u64,
}

impl VariableNeighborhoodDescent {
    /// Cria um VND padrão com a sequência: Insertion -> Swap -> Removal
    pub fn default(dimension: SearchDimension) -> Self {
        let base_config = LocalSearchConfig {
            dimension,
            strategy: SearchStrategy::FirstImprovement,
            neighborhood: NeighborhoodType::Insertion,
            sampling_size: 5000,
            max_iterations: 100,
            max_time_secs: 150,
        };
        Self {
            neighborhoods: vec![
                ConfigurableLocalSearch { config: LocalSearchConfig { neighborhood: NeighborhoodType::Insertion, ..base_config.clone() } },
                ConfigurableLocalSearch { config: LocalSearchConfig { neighborhood: NeighborhoodType::Swap, ..base_config.clone() } },
                ConfigurableLocalSearch { config: LocalSearchConfig { neighborhood: NeighborhoodType::Removal, ..base_config.clone() } },
            ],
            max_time_secs: 150,
        }
    }
}

impl LocalSearchAlgorithm for VariableNeighborhoodDescent {
    fn name(&self) -> String {
        let n_names: Vec<String> = self.neighborhoods.iter().map(|n| format!("{:?}", n.config.neighborhood)).collect();
        let dim = self.neighborhoods.first().map(|n| n.config.dimension).unwrap_or(SearchDimension::Orders);
        format!("VND [{}] on {:?}", n_names.join(" -> "), dim)
    }

    fn refine(&self, solution: &mut ChallengeSolution, data: &ProblemData, seed: u64) -> bool {
        let start_time = Instant::now();
        let mut k = 0;
        let mut global_improvement = false;

        while k < self.neighborhoods.len() {
            if start_time.elapsed().as_secs() >= self.max_time_secs {
                break;
            }

            let search_method = &self.neighborhoods[k];
            
            // Tenta melhorar na vizinhança atual (k)
            let improved = search_method.refine(solution, data, seed + k as u64);

            if improved {
                global_improvement = true;
                k = 0; // Se melhorou, volta para a primeira vizinhança
            } else {
                k += 1; // Se não melhorou, tenta a próxima vizinhança
            }
        }

        global_improvement
    }
}
