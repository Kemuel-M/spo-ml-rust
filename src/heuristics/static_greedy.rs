use crate::solution::{ChallengeSolution, ProblemData};
use super::ConstructiveAlgorithm;

pub struct StaticGreedy;

impl ConstructiveAlgorithm for StaticGreedy {
    fn name(&self) -> String { "Static Greedy".to_string() }

    fn construct(&self, data: &ProblemData, _seed: u64, _weights: Option<(&[f64], crate::local_searchs::SearchDimension)>) -> ChallengeSolution {
        let mut scored_orders = Vec::new();
        for idx in 0..data.orders.len() {
            let total = data.order_total_items[idx];
            let req = &data.order_required_aisles[idx];
            let score = if req.is_empty() { 0.0 } else { total as f64 / req.len() as f64 };
            scored_orders.push((idx, score));
        }
        scored_orders.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        let mut solution = ChallengeSolution::new(data.orders.len(), data.aisles.len());
        let mut curr_items = 0;
        let mut best_obj = 0.0;
        
        use crate::evaluator::core_stock::GlobalStock;
        let mut stock = GlobalStock::new(data.n_items);

        let mut unopened_req = Vec::with_capacity(32);
        for (idx, _) in scored_orders {
            let total = data.order_total_items[idx];
            if curr_items + total > data.wave_size_ub { continue; }
            let req = &data.order_required_aisles[idx];
            
            unopened_req.clear();
            for &a in req {
                if !solution.aisles.contains(a) { unopened_req.push(a); }
            }
            
            if !stock.can_fulfill_order_with_aisles(idx, &unopened_req, data) { continue; }
            
            if curr_items >= data.wave_size_lb {
                let new_ac = solution.aisles.count_ones(..) + unopened_req.len();
                let new_obj = (curr_items + total) as f64 / new_ac as f64;
                if new_obj <= best_obj { continue; }
            }
            
            curr_items += total;
            solution.orders.insert(idx);
            for &a in &unopened_req {
                solution.aisles.insert(a);
                stock.add_aisle_sparse(a, data);
            }
            stock.remove_order_sparse(idx, data);
            
            best_obj = curr_items as f64 / solution.aisles.count_ones(..) as f64;
        }
        solution
    }
}
