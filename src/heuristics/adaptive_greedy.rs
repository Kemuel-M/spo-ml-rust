use crate::solution::{ChallengeSolution, ProblemData};
use super::ConstructiveAlgorithm;

pub struct AdaptiveGreedy;

impl ConstructiveAlgorithm for AdaptiveGreedy {
    fn name(&self) -> String { "Adaptive Greedy".to_string() }

    fn construct(&self, data: &ProblemData, _seed: u64) -> ChallengeSolution {
        let n_orders = data.orders.len();
        let mut solution = ChallengeSolution::new(n_orders, data.aisles.len());
        let mut surplus_stock = vec![0u32; data.n_items];
        let mut curr_items = 0;
        let mut best_obj = 0.0;
        let mut unselected = vec![true; n_orders];
        
        let mut current_new_ac: Vec<usize> = data.order_required_aisles.iter()
            .map(|req| req.len())
            .collect();

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
                    let mut can_fulfill = true;
                    for item in &data.dense_orders[idx] {
                        let mut stock_we_will_have = surplus_stock[item.id];
                        for &a in req {
                            if !solution.aisles.contains(a) {
                                stock_we_will_have += data.stock_matrix[a * data.n_items + item.id];
                            }
                        }
                        if stock_we_will_have < item.qty {
                            can_fulfill = false;
                            break;
                        }
                    }
                    
                    if can_fulfill {
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
                        for item in &data.dense_aisles[a] {
                            surplus_stock[item.id] += item.qty;
                        }
                    }
                }
                
                for item in &data.dense_orders[idx] {
                    surplus_stock[item.id] -= item.qty;
                }
                
                for &a in &newly_added_aisles {
                    for &o_idx in &data.aisle_to_orders_req[a] {
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
