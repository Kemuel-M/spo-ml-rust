use crate::solution::{ChallengeSolution, ProblemData};
use super::ConstructiveAlgorithm;
use std::cmp::Ordering;
use rand::prelude::*;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

pub struct RandomGreedy {
    pub alpha: f64,
}

impl ConstructiveAlgorithm for RandomGreedy {
    fn name(&self) -> String { format!("Random Greedy (alpha={:.2})", self.alpha) }

    fn with_alpha(&self, alpha: f64) -> Option<Box<dyn ConstructiveAlgorithm>> {
        Some(Box::new(RandomGreedy { alpha }))
    }

    fn construct(&self, data: &ProblemData, seed: u64) -> ChallengeSolution {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let n_orders = data.orders.len();
        let mut solution = ChallengeSolution::new(n_orders, data.aisles.len());
        
        use crate::evaluator::core_stock::GlobalStock;
        let mut stock = GlobalStock::new(data.n_items);
        
        let mut curr_items = 0;
        let mut current_obj = 0.0;

        let mut current_new_ac: Vec<usize> = data.order_required_aisles.iter().map(|req| req.len()).collect();

        let mut unselected: Vec<usize> = (0..n_orders).collect();
        let mut unopened_req = Vec::with_capacity(32);
        let mut candidates = Vec::with_capacity(n_orders);

        while curr_items < data.wave_size_ub {
            candidates.clear();
            let best_obj = current_obj;

            for &idx in &unselected {
                let total = data.order_total_items[idx];
                if curr_items + total > data.wave_size_ub { continue; }
                
                let req = &data.order_required_aisles[idx];
                
                unopened_req.clear();
                for &a in req {
                    if !solution.aisles.contains(a) { unopened_req.push(a); }
                }
                
                if stock.can_fulfill_order_with_aisles(idx, &unopened_req, data) {
                    let new_ac = current_new_ac[idx];
                    let score = if new_ac == 0 { f64::MAX / 2.0 + total as f64 } else { total as f64 / (new_ac as f64).powi(2) };
                    if curr_items >= data.wave_size_lb {
                        let total_ac = solution.aisles.count_ones(..) + new_ac;
                        let new_obj = (curr_items + total) as f64 / total_ac as f64;
                        if new_obj <= best_obj { continue; }
                    }
                    candidates.push((idx, score));
                }
            }

            if candidates.is_empty() { break; }
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

            let limit = (candidates.len() as f64 * self.alpha).max(1.0) as usize;
            let chosen_idx_in_candidates = if self.alpha == 0.0 { 0 } else { rng.random_range(0..limit.min(candidates.len())) };
            let (chosen_id, _) = candidates.remove(chosen_idx_in_candidates);
            
            let pos = unselected.iter().position(|&x| x == chosen_id).unwrap();
            unselected.swap_remove(pos);

            let total = data.order_total_items[chosen_id];
            let req = &data.order_required_aisles[chosen_id];

            unopened_req.clear();
            for &a in req {
                if !solution.aisles.contains(a) { unopened_req.push(a); }
            }

            solution.orders.insert(chosen_id);
            curr_items += total;
            
            for &a in &unopened_req {
                solution.aisles.insert(a);
                stock.add_aisle_dense(a, data);
                for &o_idx in &data.aisle_to_orders_req[a] {
                    current_new_ac[o_idx] = current_new_ac[o_idx].saturating_sub(1);
                }
            }
            
            stock.remove_order_sparse(chosen_id, data);
            
            current_obj = curr_items as f64 / solution.aisles.count_ones(..) as f64;
        }
        
        solution.score = current_obj;
        solution.feasible = curr_items >= data.wave_size_lb && curr_items <= data.wave_size_ub;
        solution
    }
}
