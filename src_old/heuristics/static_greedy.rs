use crate::solution::{ChallengeSolution, ProblemData};
use super::ConstructiveAlgorithm;
use super::utils;

pub struct StaticGreedy;

impl ConstructiveAlgorithm for StaticGreedy {
    fn name(&self) -> String { "Static Greedy".to_string() }

    fn construct(&self, data: &ProblemData, _seed: u64) -> ChallengeSolution {
        let mut scored_orders = Vec::new();
        for idx in 0..data.orders.len() {
            let total = data.order_total_items[idx];
            let req = &data.order_required_aisles[idx];
            let score = if req.is_empty() { 0.0 } else { total as f64 / req.len() as f64 };
            scored_orders.push((idx, score));
        }
        scored_orders.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0)) // Tie-break: menor índice primeiro
        });

        let mut solution = ChallengeSolution::new(data.orders.len(), data.aisles.len());
        let mut curr_items = 0;
        let mut best_obj = 0.0;
        let mut available_stock = data.stock_matrix.clone();

        for (idx, _) in scored_orders {
            let total = data.order_total_items[idx];
            if curr_items + total > data.wave_size_ub { continue; }
            let req = &data.order_required_aisles[idx];
            if utils::is_stock_sufficient_dense(&data.dense_orders[idx], req, &available_stock, data.n_items) {
                if curr_items >= data.wave_size_lb {
                    let new_ac = solution.aisles.count_ones(..) + req.iter().filter(|&&a| !solution.aisles.contains(a)).count();
                    let new_obj = (curr_items + total) as f64 / new_ac as f64;
                    if new_obj <= best_obj { continue; }
                }
                solution.orders.insert(idx);
                for &a in req { solution.aisles.insert(a); }
                utils::update_stock_dense(&data.dense_orders[idx], &data.item_to_aisles, &mut available_stock, &solution.aisles, data.n_items);
                curr_items += total;
                best_obj = curr_items as f64 / solution.aisles.count_ones(..) as f64;
            }
        }
        solution
    }
}
