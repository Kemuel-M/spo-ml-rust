use crate::solution::{ChallengeSolution, ProblemData};
use crate::heuristics::LocalSearchAlgorithm;
use crate::evaluator::{Evaluator, StockBalanceEvaluator, Move};
use crate::local_searchs::LocalSearchConfig;
use std::collections::HashMap;
use rayon::prelude::*;

pub struct TabuSearch {
    pub config: LocalSearchConfig,
    pub tenure: usize,
    pub neighborhood_engine: Box<dyn crate::local_searchs::neighborhood::Neighborhood>,
}

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
enum TabuKey { Order(usize), Aisle(usize) }

impl LocalSearchAlgorithm for TabuSearch {
    fn name(&self) -> String { format!("Tabu Search [T={}, I={}]", self.tenure, self.config.max_iterations) }

    fn refine(&self, solution: &mut ChallengeSolution, data: &ProblemData, _seed: u64) -> bool {
        let mut evaluator: Box<dyn Evaluator> = Box::new(StockBalanceEvaluator::new(solution, data));
        solution.orders = evaluator.get_active_orders();
        solution.aisles = evaluator.get_active_aisles();

        let initial_obj = evaluator.current_objective();
        let mut best_global_obj = initial_obj;
        let mut best_solution_orders = solution.orders.clone();
        let mut best_solution_aisles = solution.aisles.clone();
        
        let mut tabu_list: HashMap<TabuKey, usize> = HashMap::new(); 
        let mut current_iter = 0;
        let start_time = std::time::Instant::now();

        while current_iter < self.config.max_iterations {
            if start_time.elapsed().as_secs() >= self.config.max_time_secs {
                break;
            }
            current_iter += 1;
            
            // Cleanup tabu list
            tabu_list.retain(|_, &mut expiry| expiry > current_iter);
            
            let moves_to_eval = self.neighborhood_engine.generate_moves(solution, data, self.config.sampling_size);
            if moves_to_eval.is_empty() { break; }

            // Parallel evaluation of the entire neighborhood
            let move_results: Vec<(Move, f64, bool)> = moves_to_eval.into_par_iter()
                .map_init(
                    || evaluator.clone_box(),
                    |local_eval, mv| {
                        if local_eval.validate_move(&mv, data) {
                            let current_obj = local_eval.current_objective();
                            match mv {
                                Move::OrderSwap(o, i) => local_eval.try_apply_order_swap(o, i, data),
                                Move::OrderInsertion(i) => local_eval.try_apply_order_add(i, data),
                                Move::OrderRemoval(o) => local_eval.try_apply_order_remove(o, data),
                                Move::AisleSwap(o, i) => local_eval.try_apply_aisle_swap(o, i, data),
                                Move::AisleInsertion(i) => local_eval.try_apply_aisle_add(i, data),
                                Move::AisleRemoval(o) => local_eval.try_apply_aisle_remove(o, data),
                            };
                            let new_obj = local_eval.current_objective();
                            let delta = new_obj - current_obj;
                            local_eval.rollback();
                            
                            // Determine tabu status
                            let is_tabu = match mv {
                                Move::OrderSwap(o, i) => tabu_list.contains_key(&TabuKey::Order(o)) || tabu_list.contains_key(&TabuKey::Order(i)),
                                Move::OrderInsertion(i) => tabu_list.contains_key(&TabuKey::Order(i)),
                                Move::OrderRemoval(o) => tabu_list.contains_key(&TabuKey::Order(o)),
                                Move::AisleSwap(o, i) => tabu_list.contains_key(&TabuKey::Aisle(o)) || tabu_list.contains_key(&TabuKey::Aisle(i)),
                                Move::AisleInsertion(i) => tabu_list.contains_key(&TabuKey::Aisle(i)),
                                Move::AisleRemoval(o) => tabu_list.contains_key(&TabuKey::Aisle(o)),
                            };

                            return Some((mv, delta, is_tabu));
                        }
                        None
                    }
                )
                .filter_map(|x| x)
                .collect();

            if move_results.is_empty() { break; }

            // Select best allowed move (non-tabu or aspirated)
            let mut best_allowed: Option<(Move, f64)> = None;
            let mut best_any: Option<(Move, f64)> = None;

            for (mv, delta, is_tabu) in move_results {
                let abs_obj = evaluator.current_objective() + delta;
                let aspirated = abs_obj > best_global_obj + 1e-6;

                if !is_tabu || aspirated {
                    if best_allowed.is_none() || delta > best_allowed.as_ref().unwrap().1 {
                        best_allowed = Some((mv, delta));
                    }
                }
                
                if best_any.is_none() || delta > best_any.as_ref().unwrap().1 {
                    best_any = Some((mv, delta));
                }
            }

            let chosen_move = best_allowed.or(best_any);
            
            if let Some((mv, _)) = chosen_move {
                match mv {
                    Move::OrderSwap(o, i) => evaluator.try_apply_order_swap(o, i, data),
                    Move::OrderInsertion(i) => evaluator.try_apply_order_add(i, data),
                    Move::OrderRemoval(o) => evaluator.try_apply_order_remove(o, data),
                    Move::AisleSwap(o, i) => evaluator.try_apply_aisle_swap(o, i, data),
                    Move::AisleInsertion(i) => evaluator.try_apply_aisle_add(i, data),
                    Move::AisleRemoval(o) => evaluator.try_apply_aisle_remove(o, data),
                };
                evaluator.commit();
                
                solution.orders = evaluator.get_active_orders();
                solution.aisles = evaluator.get_active_aisles();
                
                let current_best_in_iter = evaluator.current_objective();
                if current_best_in_iter > best_global_obj + 1e-6 {
                    best_global_obj = current_best_in_iter;
                    best_solution_orders = solution.orders.clone();
                    best_solution_aisles = solution.aisles.clone();
                }

                // Update tabu list
                match mv {
                    Move::OrderSwap(o, i) => { tabu_list.insert(TabuKey::Order(o), current_iter + self.tenure); tabu_list.insert(TabuKey::Order(i), current_iter + self.tenure); },
                    Move::OrderInsertion(i) => { tabu_list.insert(TabuKey::Order(i), current_iter + self.tenure); },
                    Move::OrderRemoval(o) => { tabu_list.insert(TabuKey::Order(o), current_iter + self.tenure); },
                    Move::AisleSwap(o, i) => { tabu_list.insert(TabuKey::Aisle(o), current_iter + self.tenure); tabu_list.insert(TabuKey::Aisle(i), current_iter + self.tenure); },
                    Move::AisleInsertion(i) => { tabu_list.insert(TabuKey::Aisle(i), current_iter + self.tenure); },
                    Move::AisleRemoval(o) => { tabu_list.insert(TabuKey::Aisle(o), current_iter + self.tenure); },
                }
            } else { break; }
        }

        solution.orders = best_solution_orders;
        solution.aisles = best_solution_aisles;
        best_global_obj > initial_obj + 1e-6
    }
}
