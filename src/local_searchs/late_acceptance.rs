use crate::solution::{ChallengeSolution, ProblemData};
use crate::heuristics::LocalSearchAlgorithm;
use crate::evaluator::{Evaluator, StockBalanceEvaluator, Move};
use crate::local_searchs::LocalSearchConfig;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand::prelude::*;

pub struct LateAcceptanceHillClimbing {
    pub config: LocalSearchConfig,
    pub list_size: usize,
}

impl LocalSearchAlgorithm for LateAcceptanceHillClimbing {
    fn name(&self) -> String {
        format!("LAHC [L={}, I={} on {:?}]", self.list_size, self.config.max_iterations, self.config.dimension)
    }

    fn refine(&self, solution: &mut ChallengeSolution, data: &ProblemData, seed: u64) -> bool {
        let mut evaluator: Box<dyn Evaluator> = Box::new(StockBalanceEvaluator::new(solution, data));
        solution.orders = evaluator.get_active_orders();
        solution.aisles = evaluator.get_active_aisles();

        let initial_obj = evaluator.current_objective();
        let mut cost_list = vec![initial_obj; self.list_size];
        let mut list_idx = 0;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        
        let mut best_global_obj = initial_obj;
        let mut best_solution_orders = solution.orders.clone();
        let mut best_solution_aisles = solution.aisles.clone();

        let n_orders = data.orders.len();
        let n_aisles = data.aisles.len();
        let start_time = std::time::Instant::now();

        for _ in 0..self.config.max_iterations {
            if start_time.elapsed().as_secs() >= self.config.max_time_secs {
                break;
            }
            let mut mv = None;
            
            // Try to find a valid random move (limit to 50 attempts to avoid infinite loops)
            for _ in 0..50 {
                let candidate_mv = match self.config.dimension {
                    crate::local_searchs::SearchDimension::Orders => {
                        let orders_in: Vec<usize> = solution.orders.ones().collect();
                        if orders_in.is_empty() { 
                            Move::OrderInsertion(rng.random_range(0..n_orders))
                        } else {
                            match self.config.neighborhood {
                                crate::local_searchs::NeighborhoodType::Swap => {
                                    let rem = *orders_in.choose(&mut rng).unwrap();
                                    let mut add = rng.random_range(0..n_orders);
                                    while solution.orders.contains(add) { add = rng.random_range(0..n_orders); }
                                    Move::OrderSwap(rem, add)
                                },
                                crate::local_searchs::NeighborhoodType::Insertion => {
                                    let mut add = rng.random_range(0..n_orders);
                                    while solution.orders.contains(add) { add = rng.random_range(0..n_orders); }
                                    Move::OrderInsertion(add)
                                },
                                crate::local_searchs::NeighborhoodType::Removal => {
                                    Move::OrderRemoval(*orders_in.choose(&mut rng).unwrap())
                                }
                            }
                        }
                    },
                    crate::local_searchs::SearchDimension::Aisles => {
                        let aisles_in: Vec<usize> = solution.aisles.ones().collect();
                        if aisles_in.is_empty() {
                            Move::AisleInsertion(rng.random_range(0..n_aisles))
                        } else {
                            match self.config.neighborhood {
                                crate::local_searchs::NeighborhoodType::Swap => {
                                    let rem = *aisles_in.choose(&mut rng).unwrap();
                                    let mut add = rng.random_range(0..n_aisles);
                                    while solution.aisles.contains(add) { add = rng.random_range(0..n_aisles); }
                                    Move::AisleSwap(rem, add)
                                },
                                crate::local_searchs::NeighborhoodType::Insertion => {
                                    let mut add = rng.random_range(0..n_aisles);
                                    while solution.aisles.contains(add) { add = rng.random_range(0..n_aisles); }
                                    Move::AisleInsertion(add)
                                },
                                crate::local_searchs::NeighborhoodType::Removal => {
                                    Move::AisleRemoval(*aisles_in.choose(&mut rng).unwrap())
                                }
                            }
                        }
                    }
                };

                if evaluator.validate_move(&candidate_mv, data) {
                    mv = Some(candidate_mv);
                    break;
                }
            }

            if let Some(valid_mv) = mv {
                let current_obj_before_move = evaluator.current_objective();
                match valid_mv {
                    Move::OrderSwap(o, i) => evaluator.try_apply_order_swap(o, i, data),
                    Move::OrderInsertion(i) => evaluator.try_apply_order_add(i, data),
                    Move::OrderRemoval(o) => evaluator.try_apply_order_remove(o, data),
                    Move::AisleSwap(o, i) => evaluator.try_apply_aisle_swap(o, i, data),
                    Move::AisleInsertion(i) => evaluator.try_apply_aisle_add(i, data),
                    Move::AisleRemoval(o) => evaluator.try_apply_aisle_remove(o, data),
                };

                let candidate_obj = evaluator.current_objective();
                let prev_obj = cost_list[list_idx];

                if candidate_obj >= prev_obj || candidate_obj >= current_obj_before_move {
                    evaluator.commit();
                    solution.orders = evaluator.get_active_orders();
                    solution.aisles = evaluator.get_active_aisles();
                    
                    if candidate_obj > best_global_obj + 1e-6 {
                        best_global_obj = candidate_obj;
                        best_solution_orders = solution.orders.clone();
                        best_solution_aisles = solution.aisles.clone();
                    }
                } else {
                    evaluator.rollback();
                }
            }
            
            cost_list[list_idx] = evaluator.current_objective();
            list_idx = (list_idx + 1) % self.list_size;
        }

        solution.orders = best_solution_orders;
        solution.aisles = best_solution_aisles;
        best_global_obj > initial_obj + 1e-6
    }
}
