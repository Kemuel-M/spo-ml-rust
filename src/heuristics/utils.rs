use crate::solution::{ChallengeSolution, ProblemData, DenseItem};
use std::collections::{HashMap, HashSet};
use fixedbitset::FixedBitSet;

pub fn is_stock_sufficient_dense(order_items: &[DenseItem], req_aisles: &[usize], available_stock: &[u32], n_items: usize) -> bool {
    for item in order_items {
        if item.id >= n_items { continue; }
        let mut total = 0;
        for &a in req_aisles {
            total += available_stock[a * n_items + item.id];
        }
        if total < item.qty { return false; }
    }
    true
}

pub fn update_stock_dense(order_items: &[DenseItem], item_to_aisles: &[Vec<usize>], available_stock: &mut [u32], solution_aisles: &FixedBitSet, n_items: usize) {
    for item in order_items {
        if item.id >= n_items { continue; }
        let mut qty = item.qty;
        let aisles = &item_to_aisles[item.id];
        for &a_idx in aisles.iter().filter(|&&a| solution_aisles.contains(a)) {
            if qty == 0 { break; }
            let s = &mut available_stock[a_idx * n_items + item.id];
            let take = std::cmp::min(qty, *s);
            *s -= take;
            qty -= take;
        }
        if qty > 0 {
            // This should not happen if is_stock_sufficient_dense was true
            // unless solution_aisles doesn't contain the req aisles.
        }
    }
}

pub fn compute_raw_objective(solution: &ChallengeSolution, data: &ProblemData) -> f64 {
    let ac = solution.aisles.count_ones(..);
    if ac == 0 || solution.orders.is_empty() { return 0.0; }
    let items: u32 = solution.orders.ones().map(|idx| data.order_total_items[idx]).sum();
    items as f64 / ac as f64
}

pub fn compute_objective(solution: &ChallengeSolution, data: &ProblemData) -> f64 {
    let ac = solution.aisles.count_ones(..);
    if ac == 0 || solution.orders.is_empty() { return 0.0; }
    if !is_solution_stock_valid(solution, data) { return 0.0; }
    let items: u32 = solution.orders.ones().map(|idx| data.order_total_items[idx]).sum();
    let raw = items as f64 / ac as f64;
    let mut penalty = 0.0;
    if items < data.wave_size_lb { penalty = (data.wave_size_lb - items) as f64 * 5.0; }
    else if items > data.wave_size_ub { penalty = (items - data.wave_size_ub) as f64 * 5.0; }
    (raw - penalty).max(0.0)
}

pub fn map_item_to_aisles(data: &ProblemData) -> HashMap<usize, HashSet<usize>> {
    let mut map = HashMap::new();
    for (a_idx, items) in data.aisles.iter().enumerate() {
        for &it_id in items.keys() { map.entry(it_id).or_insert_with(HashSet::new).insert(a_idx); }
    }
    map
}

pub fn get_required_aisles_with_stock(order_items: &HashMap<usize, u32>, data: &ProblemData, _item_to_aisles: &HashMap<usize, HashSet<usize>>) -> HashSet<usize> {
    let mut remaining = order_items.clone();
    remaining.retain(|_, &mut q| q > 0);
    let mut selected = HashSet::new();
    while !remaining.is_empty() {
        let mut best_aisle = None;
        let mut max_cov = 0;
        let mut candidates = HashSet::new();
        let mut sorted_keys: Vec<_> = remaining.keys().cloned().collect();
        sorted_keys.sort();
        for it_id in sorted_keys {
            for a_idx in data.item_locations_bits[it_id].ones() {
                if !selected.contains(&a_idx) { candidates.insert(a_idx); }
            }
        }
        let mut sorted_candidates: Vec<_> = candidates.into_iter().collect();
        sorted_candidates.sort();
        for a_idx in sorted_candidates {
            let mut cov = 0;
            for (&it_id, &dem) in &remaining {
                cov += std::cmp::min(dem, data.stock_matrix[a_idx * data.n_items + it_id]);
            }
            if cov > max_cov { max_cov = cov; best_aisle = Some(a_idx); }
        }
        if let Some(a_idx) = best_aisle {
            selected.insert(a_idx);
            let mut to_rem = Vec::new();
            for (&it_id, dem) in remaining.iter_mut() {
                let stk = data.stock_matrix[a_idx * data.n_items + it_id];
                if stk >= *dem { to_rem.push(it_id); } else { *dem -= stk; }
            }
            for it in to_rem { remaining.remove(&it); }
        } else { break; }
    }
    selected
}

pub fn is_stock_sufficient(order_items: &HashMap<usize, u32>, aisles: &HashSet<usize>, available_stock: &[HashMap<usize, u32>]) -> bool {
    for (&it_id, &req) in order_items {
        let total: u32 = aisles.iter().filter_map(|&a| available_stock[a].get(&it_id)).sum();
        if total < req { return false; }
    }
    true
}

pub fn is_solution_stock_valid(solution: &ChallengeSolution, data: &ProblemData) -> bool {
    let mut demand = HashMap::new();
    for o_idx in solution.orders.ones() {
        for (&it, &q) in &data.orders[o_idx] { *demand.entry(it).or_insert(0) += q; }
    }
    for (it, req) in demand {
        let mut avail = 0;
        for a_idx in solution.aisles.ones() {
            avail += data.stock_matrix[a_idx * data.n_items + it];
            if avail >= req { break; }
        }
        if avail < req { return false; }
    }
    true
}

pub fn is_feasible(solution: &ChallengeSolution, data: &ProblemData) -> bool {
    if !is_solution_stock_valid(solution, data) { return false; }
    let items: u32 = solution.orders.ones().map(|idx| data.order_total_items[idx]).sum();
    items >= data.wave_size_lb && items <= data.wave_size_ub
}

pub fn guarantee_feasibility(solution: &mut ChallengeSolution, data: &ProblemData) {
    use crate::evaluator::{StockBalanceEvaluator, Evaluator};
    let mut eval = StockBalanceEvaluator::new(solution, data);
    if !eval.is_stock_valid() { eval.sync_aisles_from_orders(data); }
    if eval.current_total_items() < data.wave_size_lb {
        eval.sync_orders_from_aisles(data);
        if eval.current_total_items() < data.wave_size_lb {
            for &o_idx in &data.orders_sorted_by_size {
                if eval.current_total_items() >= data.wave_size_lb { break; }
                if !eval.get_active_orders().contains(o_idx) { eval.try_apply_order_add(o_idx, data); }
            }
        }
    }
    if eval.current_total_items() > data.wave_size_ub {
        let mut orders: Vec<usize> = eval.get_active_orders().ones().collect();
        orders.sort_unstable_by(|&a, &b| data.order_total_items[a].cmp(&data.order_total_items[b]));
        for o in orders {
            if eval.current_total_items() <= data.wave_size_ub { break; }
            eval.try_apply_order_remove(o, data);
        }
    }
    solution.orders = eval.get_active_orders();
    solution.aisles = eval.get_active_aisles();
}
