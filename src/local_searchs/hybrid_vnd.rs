use crate::solution::{ChallengeSolution, ProblemData};
use crate::heuristics::LocalSearchAlgorithm;
use crate::local_searchs::{ConfigurableLocalSearch, SearchStrategy, NeighborhoodType, LocalSearchConfig, SearchDimension};
use std::time::Instant;
use log::debug;

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
        debug!("HVND Iniciado: score base={:.4}", solution.score);
        
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
                debug!("HVND: melhoria encontrada na vizinhança {}, novo score={:.4}", k, solution.score);
                k = 0; // Volta para o início se melhorou
            } else {
                k += 1;
            }
        }

        global_improvement
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::read_input;
    use std::time::Instant;

    fn get_test_data() -> ProblemData {
        read_input("datasets/x/instance_0010.txt", false).expect("Failed to load test instance")
    }

    fn create_initial_solution(data: &ProblemData) -> ChallengeSolution {
        let mut sol = ChallengeSolution::new(data.orders.len(), data.aisles.len());
        // Cria uma solucao viavel basica (primeiros 5 pedidos)
        for i in 0..5 {
            sol.orders.insert(i);
            for &a in data.order_required_aisles[i].iter() {
                sol.aisles.insert(a);
            }
        }
        
        let eval = StockBalanceEvaluator::new(&sol, data);
        sol.score = eval.current_objective();
        sol
    }

    #[test]
    fn test_hvnd_full_execution_and_timing() {
        let data = get_test_data();
        let mut sol = create_initial_solution(&data);
        let initial_score = sol.score;

        let mut hvnd = HybridVND::default();
        hvnd.max_time_secs = 2; // Limite baixo para o teste nao demorar eternamente se houver infinitos updates
        
        let start = Instant::now();
        let improved = hvnd.refine(&mut sol, &data, 42);
        let duration = start.elapsed();

        assert!(duration.as_secs() <= 3, "HVND demorou demais e furou o limite de tempo do teste");

        if improved {
            assert!(sol.score > initial_score, "Se a busca melhorou, o score deve ter subido");
        } else {
            assert_eq!(sol.score, initial_score, "Se nao melhorou, o score tem que ser mantido");
        }
    }

    #[test]
    fn test_hvnd_neighborhoods_individually_timing() {
        let data = get_test_data();
        let sol = create_initial_solution(&data);
        
        let mut hvnd = HybridVND::default();
        
        for nbh in hvnd.neighborhoods.iter_mut() {
            nbh.config.max_time_secs = 1; // Força limite de tempo em cada vizinhança
            nbh.config.max_iterations = 5; // Evita loop infinito em testes se houver falsos positivos
            
            let mut sol_clone = sol.clone();
            let start = Instant::now();
            let improved = nbh.refine(&mut sol_clone, &data, 42);
            let duration = start.elapsed();
            
            assert!(duration.as_secs() <= 2, "Vizinhança extrapolou limite de tempo!");
            if improved {
                assert!(sol_clone.score > sol.score, "Melhoria reportada, mas score nao subiu");
            }
        }
    }
}
