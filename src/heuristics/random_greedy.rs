use crate::solution::{ChallengeSolution, ProblemData};
use super::ConstructiveAlgorithm;
use rand::prelude::*;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

pub struct RandomGreedy {
    pub alpha: f64,
}

use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Clone, Copy)]
struct Candidate {
    id: usize,
    score: f64,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool { self.id == other.id }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.score.partial_cmp(&other.score)
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
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

        let mut current_new_ac: Vec<usize> = data.order_initial_aisles_count.clone();

        let mut heap = BinaryHeap::with_capacity(n_orders);
        for i in 0..n_orders {
            let total = data.order_total_items[i];
            let new_ac = current_new_ac[i];
            let score = if new_ac == 0 { f64::MAX / 2.0 + total as f64 } else { total as f64 / (new_ac as f64).powi(2) };
            heap.push(Candidate { id: i, score });
        }

        let mut unopened_req = Vec::with_capacity(32);
        
        while curr_items < data.wave_size_ub && !heap.is_empty() {
            let mut rcl = Vec::with_capacity(32);
            
            while rcl.len() < 20 && !heap.is_empty() {
                let cand = heap.pop().unwrap();
                let total = data.order_total_items[cand.id];
                
                if curr_items + total > data.wave_size_ub { continue; }
                
                let actual_new_ac = current_new_ac[cand.id];
                let actual_score = if actual_new_ac == 0 { f64::MAX / 2.0 + total as f64 } else { total as f64 / (actual_new_ac as f64).powi(2) };
                
                if (cand.score - actual_score).abs() > 1e-6 {
                    heap.push(Candidate { id: cand.id, score: actual_score });
                    continue;
                }

                unopened_req.clear();
                for &a in &data.order_required_aisles[cand.id] {
                    if !solution.aisles.contains(a) { unopened_req.push(a); }
                }
                
                if stock.can_fulfill_order_with_aisles(cand.id, &unopened_req, data) {
                    rcl.push(cand);
                }
            }

            if rcl.is_empty() { break; }

            let limit = (rcl.len() as f64 * self.alpha).max(1.0) as usize;
            let chosen_idx = if self.alpha == 0.0 { 0 } else { rng.random_range(0..limit.min(rcl.len())) };
            let chosen = rcl.remove(chosen_idx);
            
            for remaining in rcl { heap.push(remaining); }

            let idx = chosen.id;
            let total = data.order_total_items[idx];
            let req = &data.order_required_aisles[idx];

            if curr_items >= data.wave_size_lb {
                let new_ac_count = current_new_ac[idx];
                let total_ac = solution.aisles.count_ones(..) + new_ac_count;
                let new_obj = (curr_items + total) as f64 / total_ac as f64;
                if new_obj <= current_obj {
                    continue; 
                }
            }

            solution.orders.insert(idx);
            curr_items += total;
            
            unopened_req.clear();
            for &a in req {
                if !solution.aisles.contains(a) { unopened_req.push(a); }
            }

            for &a in &unopened_req {
                solution.aisles.insert(a);
                stock.add_aisle_sparse(a, data);
                for &o_idx in &data.aisle_to_orders_req[a] {
                    current_new_ac[o_idx] = current_new_ac[o_idx].saturating_sub(1);
                }
            }
            
            stock.remove_order_sparse(idx, data);
            
            current_obj = curr_items as f64 / solution.aisles.count_ones(..) as f64;
        }
        
        solution.score = current_obj;
        solution.feasible = curr_items >= data.wave_size_lb && curr_items <= data.wave_size_ub;
        solution
    }
}
