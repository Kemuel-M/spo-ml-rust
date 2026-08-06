use crate::solution::{ChallengeSolution, ProblemData};
use crate::heuristics::LocalSearchAlgorithm;
use crate::evaluator::{Evaluator, StockBalanceEvaluator, Move};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use super::{SearchStrategy, LocalSearchConfig};
use log::trace;

pub struct ConfigurableLocalSearch {
    pub config: LocalSearchConfig,
    pub neighborhood_engine: Box<dyn crate::local_searchs::neighborhood::Neighborhood>,
}

impl LocalSearchAlgorithm for ConfigurableLocalSearch {
    fn name(&self) -> String {
        format!("{:?} using {:?} on {:?}", self.config.strategy, self.config.neighborhood, self.config.dimension)
    }

    fn refine(&self, solution: &mut ChallengeSolution, data: &ProblemData, seed: u64) -> bool {
        let mut evaluator: Box<dyn Evaluator> = Box::new(StockBalanceEvaluator::new(solution, data));
        
        // Sincroniza a solução inicial com o estado corrigido do evaluator
        solution.orders = evaluator.get_active_orders();
        solution.aisles = evaluator.get_active_aisles();
        solution.score = evaluator.current_objective();

        let initial_obj = evaluator.current_objective();
        let backup_solution = solution.clone();

        let mut global_improvement = false;
        let mut improved_iter = true;
        let mut iter_count = 0;
        let start_time = std::time::Instant::now();
        
        while improved_iter && iter_count < self.config.max_iterations {
            if start_time.elapsed().as_secs() >= self.config.max_time_secs {
                break;
            }
            iter_count += 1;
            improved_iter = false;
            
            match self.config.strategy {
                SearchStrategy::FirstImprovement => {
                    trace!("  [{}] FirstImprovement Iter: {}", self.name(), iter_count);
                    if self.run_first_improvement(&mut *evaluator, solution, data, seed + iter_count as u64) {
                        improved_iter = true;
                        global_improvement = true;
                    }
                },
                SearchStrategy::BestImprovement => {
                    if self.run_best_improvement(&mut *evaluator, solution, data) {
                        improved_iter = true;
                        global_improvement = true;
                    }
                },
            }
        }

        if evaluator.current_objective() < initial_obj - 1e-6 {
            *solution = backup_solution;
            return false;
        }
        
        // Garante que o score final esteja refletido
        solution.score = evaluator.current_objective();

        global_improvement
    }
}

impl ConfigurableLocalSearch {
    fn run_first_improvement(
        &self, 
        evaluator: &mut dyn Evaluator, 
        solution: &mut ChallengeSolution, 
        data: &ProblemData,
        seed: u64
    ) -> bool {
        let current_obj = evaluator.current_objective();
        let mut moves = self.neighborhood_engine.generate_moves(solution, data, self.config.sampling_size);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        moves.shuffle(&mut rng);

        for mv in moves {
            if evaluator.validate_move(&mv, data) {
                if self.apply_move(evaluator, solution, data, mv, current_obj) {
                    trace!("    Melhoria encontrada: {:.4} via {:?}", evaluator.current_objective(), mv);
                    return true;
                }
            }
        }
        false
    }

    fn run_best_improvement(
        &self,
        evaluator: &mut dyn Evaluator,
        solution: &mut ChallengeSolution,
        data: &ProblemData
    ) -> bool {
        let current_obj = evaluator.current_objective();
        let moves = self.neighborhood_engine.generate_moves(solution, data, self.config.sampling_size);

        use rayon::prelude::*;
        
        let best_move_found = moves.into_par_iter()
            .map_init(
                || evaluator.clone_box(), 
                |local_eval, mv| {
                        if local_eval.validate_move(&mv, data) {
                            match mv {
                                Move::OrderSwap(o, i) => local_eval.try_apply_order_swap(o, i, data),
                                Move::OrderInsertion(i) => local_eval.try_apply_order_add(i, data),
                                Move::OrderRemoval(o) => local_eval.try_apply_order_remove(o, data),
                                Move::AisleSwap(o, i) => local_eval.try_apply_aisle_swap(o, i, data),
                                Move::AisleInsertion(i) => local_eval.try_apply_aisle_add(i, data),
                                Move::AisleRemoval(o) => local_eval.try_apply_aisle_remove(o, data),
                            };
                            let new_obj = local_eval.current_objective();
                            local_eval.rollback();
                            return Some((mv, new_obj));
                        }
                    None
                }
            )
            .filter_map(|x| x)
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        if let Some((mv, best_obj)) = best_move_found {
            if best_obj > current_obj + 1e-6 {
                return self.apply_move(evaluator, solution, data, mv, current_obj);
            }
        }
        false
    }

    fn apply_move(
        &self, 
        evaluator: &mut dyn Evaluator, 
        solution: &mut ChallengeSolution, 
        data: &ProblemData, 
        mv: Move, 
        current_obj: f64
    ) -> bool {
        match mv {
            Move::OrderSwap(o, i) => evaluator.try_apply_order_swap(o, i, data),
            Move::OrderInsertion(i) => evaluator.try_apply_order_add(i, data),
            Move::OrderRemoval(o) => evaluator.try_apply_order_remove(o, data),
            Move::AisleSwap(o, i) => evaluator.try_apply_aisle_swap(o, i, data),
            Move::AisleInsertion(i) => evaluator.try_apply_aisle_add(i, data),
            Move::AisleRemoval(o) => evaluator.try_apply_aisle_remove(o, data),
        };

        let new_obj = evaluator.current_objective();
        if new_obj > current_obj + 1e-6 {
            evaluator.commit();
            solution.orders = evaluator.get_active_orders();
            solution.aisles = evaluator.get_active_aisles();
            solution.score = new_obj;
            return true;
        }
        evaluator.rollback();
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::read_input;
    use crate::local_searchs::{NeighborhoodType, SearchDimension};
    use std::time::Instant;

    fn get_test_data() -> ProblemData {
        read_input("datasets/x/instance_0010.txt", false).expect("Failed to load test instance")
    }

    fn create_initial_solution(data: &ProblemData) -> ChallengeSolution {
        let mut sol = ChallengeSolution::new(data.orders.len(), data.aisles.len());
        // Preenche de forma viavel (primeiros 5 pedidos)
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
    fn test_configurable_first_improvement_orders_insertion() {
        let data = get_test_data();
        let mut sol = create_initial_solution(&data);
        let initial_score = sol.score;

        let config = LocalSearchConfig {
            dimension: SearchDimension::Orders,
            strategy: SearchStrategy::FirstImprovement,
            neighborhood: NeighborhoodType::Insertion,
            sampling_size: 100,
            max_iterations: 1,
            max_time_secs: 2,
        };

        let engine = crate::local_searchs::neighborhood::build_neighborhood(
            SearchDimension::Orders, 
            NeighborhoodType::Insertion
        );

        let cls = ConfigurableLocalSearch {
            config,
            neighborhood_engine: engine,
        };

        let start = Instant::now();
        let improved = cls.refine(&mut sol, &data, 123);
        let duration = start.elapsed();

        println!("Time taken for Order Insertion FI: {:?}", duration);
        assert!(duration.as_secs() <= 2, "Order insertion was too slow");

        if improved {
            assert!(sol.score > initial_score, "Score should increase on improvement");
        }
    }
}
