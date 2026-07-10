use crate::solution::{ChallengeSolution, ProblemData};
use crate::heuristics::LocalSearchAlgorithm;
use crate::evaluator::{Evaluator, StockBalanceEvaluator, Move};
use crate::local_searchs::{LocalSearchConfig, SearchDimension, NeighborhoodType};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand::prelude::*;

pub struct HybridLAHC {
    pub config: LocalSearchConfig,
    pub list_size: usize,
    pub neighborhoods: Vec<(SearchDimension, Box<dyn crate::local_searchs::neighborhood::Neighborhood>)>,
}

impl HybridLAHC {
    pub fn new(config: LocalSearchConfig, list_size: usize) -> Self {
        let mut neighborhoods: Vec<(SearchDimension, Box<dyn crate::local_searchs::neighborhood::Neighborhood>)> = Vec::new();
        
        // Add neighborhoods for both dimensions
        neighborhoods.push((SearchDimension::Orders, crate::local_searchs::neighborhood::build_neighborhood(SearchDimension::Orders, NeighborhoodType::Insertion)));
        neighborhoods.push((SearchDimension::Orders, crate::local_searchs::neighborhood::build_neighborhood(SearchDimension::Orders, NeighborhoodType::Swap)));
        neighborhoods.push((SearchDimension::Aisles, crate::local_searchs::neighborhood::build_neighborhood(SearchDimension::Aisles, NeighborhoodType::Removal)));
        neighborhoods.push((SearchDimension::Aisles, crate::local_searchs::neighborhood::build_neighborhood(SearchDimension::Aisles, NeighborhoodType::Swap)));
        
        Self {
            config,
            list_size,
            neighborhoods,
        }
    }
}

impl LocalSearchAlgorithm for HybridLAHC {
    fn name(&self) -> String {
        format!("H-LAHC [L={}, I={}]", self.list_size, self.config.max_iterations)
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

        let mut idle_iterations = 0;

        for _ in 0..self.config.max_iterations {
            if start_time.elapsed().as_secs() >= self.config.max_time_secs {
                break;
            }
            if idle_iterations >= 200 {
                break;
            }
            
            let mut mv = None;
            
            // Randomly select one neighborhood structure
            let nbh_idx = rng.random_range(0..self.neighborhoods.len());
            let (dim, engine) = &self.neighborhoods[nbh_idx];
            
            // Try to find a valid random move (limit to 50 attempts to avoid infinite loops)
            for _ in 0..50 {
                let candidate_mv = engine.get_random_move(solution, data, &mut rng)
                    .unwrap_or_else(|| {
                        if *dim == SearchDimension::Orders {
                            Move::OrderInsertion(rng.random_range(0..n_orders))
                        } else {
                            Move::AisleInsertion(rng.random_range(0..n_aisles))
                        }
                    });

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
                        idle_iterations = 0; // Reset early stopping counter
                    }
                } else {
                    evaluator.rollback();
                }
            }
            
            cost_list[list_idx] = evaluator.current_objective();
            list_idx = (list_idx + 1) % self.list_size;
            idle_iterations += 1;
        }

        solution.orders = best_solution_orders;
        solution.aisles = best_solution_aisles;
        best_global_obj > initial_obj + 1e-6
    }
}
