use crate::solution::{ChallengeSolution, ProblemData};
use super::ConstructiveAlgorithm;
use super::utils;

pub struct AdaptiveGreedy;

impl ConstructiveAlgorithm for AdaptiveGreedy {
    fn name(&self) -> String { "Adaptive Greedy".to_string() }

    fn construct(&self, data: &ProblemData, _seed: u64) -> ChallengeSolution {
        let n_orders = data.orders.len();
        let mut solution = ChallengeSolution::new(n_orders, data.aisles.len());
        let mut available_stock = data.stock_matrix.clone();
        let mut curr_items = 0;
        let mut best_obj = 0.0;
        let mut unselected = vec![true; n_orders];
        
        let mut current_new_ac: Vec<usize> = data.order_required_aisles.iter()
            .map(|req| req.len())
            .collect();

        let mut aisle_to_orders_req = vec![Vec::new(); data.aisles.len()];
        for (o_idx, req) in data.order_required_aisles.iter().enumerate() {
            for &a in req {
                aisle_to_orders_req[a].push(o_idx);
            }
        }

        while curr_items < data.wave_size_ub {
            let mut best_candidate = None;
            let mut max_score = -1.0;

            for idx in 0..n_orders {
                if !unselected[idx] { continue; }
                let total = data.order_total_items[idx];
                if curr_items + total > data.wave_size_ub { continue; }
                
                let new_ac = current_new_ac[idx];
                let score = if new_ac == 0 {
                    f64::MAX / 2.0 + total as f64
                } else {
                    total as f64 / new_ac as f64
                };
                
                // Deterministic tie-break: if scores are equal, the lower index wins
                // because we iterate idx from 0 to n_orders.
                if score > max_score {
                    let req = &data.order_required_aisles[idx];
                    if utils::is_stock_sufficient_dense(&data.dense_orders[idx], req, &available_stock, data.n_items) {
                        if curr_items >= data.wave_size_lb {
                            let total_ac = solution.aisles.count_ones(..) + new_ac;
                            let new_obj = (curr_items + total) as f64 / total_ac as f64;
                            if new_obj <= best_obj { continue; }
                        }
                        max_score = score;
                        best_candidate = Some(idx);
                    }
                }
            }

            if let Some(idx) = best_candidate {
                let req = &data.order_required_aisles[idx];
                let total = data.order_total_items[idx];
                
                solution.orders.insert(idx);
                let mut newly_added_aisles = Vec::new();
                for &a in req {
                    if !solution.aisles.contains(a) {
                        solution.aisles.insert(a);
                        newly_added_aisles.push(a);
                    }
                }
                
                utils::update_stock_dense(&data.dense_orders[idx], &data.item_to_aisles, &mut available_stock, &solution.aisles, data.n_items);
                
                for &a in &newly_added_aisles {
                    for &o_idx in &aisle_to_orders_req[a] {
                         current_new_ac[o_idx] = current_new_ac[o_idx].saturating_sub(1);
                    }
                }
                
                curr_items += total;
                unselected[idx] = false;
                best_obj = curr_items as f64 / solution.aisles.count_ones(..) as f64;
            } else { 
                break; 
            }
        }
        solution
    }
}
