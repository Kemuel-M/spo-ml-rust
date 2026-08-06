use crate::solution::{ChallengeSolution, ProblemData};
use crate::heuristics::LocalSearchAlgorithm;
use crate::evaluator::{Evaluator, StockBalanceEvaluator, Move};
use super::{LocalSearchConfig, NeighborhoodType, SearchDimension};

pub struct HybridHillClimbing {
    pub config: LocalSearchConfig,
    pub neighborhoods: Vec<(SearchDimension, Box<dyn crate::local_searchs::neighborhood::Neighborhood>)>,
}

impl HybridHillClimbing {
    pub fn new(config: LocalSearchConfig) -> Self {
        let mut neighborhoods: Vec<(SearchDimension, Box<dyn crate::local_searchs::neighborhood::Neighborhood>)> = Vec::new();
        
        // Add neighborhoods for both dimensions
        neighborhoods.push((SearchDimension::Orders, crate::local_searchs::neighborhood::build_neighborhood(SearchDimension::Orders, NeighborhoodType::Insertion)));
        neighborhoods.push((SearchDimension::Orders, crate::local_searchs::neighborhood::build_neighborhood(SearchDimension::Orders, NeighborhoodType::Swap)));
        neighborhoods.push((SearchDimension::Aisles, crate::local_searchs::neighborhood::build_neighborhood(SearchDimension::Aisles, NeighborhoodType::Removal)));
        neighborhoods.push((SearchDimension::Aisles, crate::local_searchs::neighborhood::build_neighborhood(SearchDimension::Aisles, NeighborhoodType::Swap)));
        
        Self {
            config,
            neighborhoods,
        }
    }
}

impl LocalSearchAlgorithm for HybridHillClimbing {
    fn name(&self) -> String {
        format!("HHC [Union Best Improvement]")
    }

    fn refine(&self, solution: &mut ChallengeSolution, data: &ProblemData, _seed: u64) -> bool {
        let mut evaluator: Box<dyn Evaluator> = Box::new(StockBalanceEvaluator::new(solution, data));
        
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
            
            let current_obj = evaluator.current_objective();
            
            // Collect moves from ALL neighborhoods
            let mut all_moves = Vec::new();
            for (_, engine) in &self.neighborhoods {
                let moves = engine.generate_moves(solution, data, self.config.sampling_size);
                all_moves.extend(moves);
            }

            use rayon::prelude::*;
            
            let best_move_found = all_moves.into_par_iter()
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
                    // Apply move
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
                        improved_iter = true;
                        global_improvement = true;
                    } else {
                        evaluator.rollback();
                    }
                }
            }
        }

        if evaluator.current_objective() < initial_obj - 1e-6 {
            *solution = backup_solution;
            return false;
        }
        
        solution.score = evaluator.current_objective();
        global_improvement
    }
}
