use crate::solution::{ChallengeSolution, ProblemData};
use crate::heuristics::ConstructiveAlgorithm;
use std::collections::HashSet;
use rand::prelude::*;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

pub struct AisleCentricRandom { pub alpha: f64 }

impl ConstructiveAlgorithm for AisleCentricRandom {
    fn name(&self) -> String { format!("AisleCentric (Random alpha={:.2})", self.alpha) }

    fn with_alpha(&self, alpha: f64) -> Option<Box<dyn ConstructiveAlgorithm>> {
        Some(Box::new(AisleCentricRandom { alpha }))
    }

    fn construct(&self, data: &ProblemData, seed: u64, weights: Option<(&[f64], crate::local_searchs::SearchDimension)>) -> ChallengeSolution {
        let n_orders = data.orders.len();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut curr_sol = ChallengeSolution::new(n_orders, data.aisles.len());
        let mut best_sol = ChallengeSolution::new(n_orders, data.aisles.len());
        let mut best_score = 0.0;
        let mut unselected: HashSet<usize> = (0..n_orders).collect();
        let mut available_aisles: HashSet<usize> = (0..data.aisles.len()).collect();
        let mut total_curr_stock = vec![0u32; data.n_items];
        let mut total_items = 0;

        let aisle_weights = if let Some((w, crate::local_searchs::SearchDimension::Aisles)) = weights { Some(w) } else { None };
        let order_weights = if let Some((w, crate::local_searchs::SearchDimension::Orders)) = weights { Some(w) } else { None };

        let mut pending_demand = vec![0u32; data.n_items];
        for oid in 0..n_orders {
            for item in &data.dense_orders[oid] {
                pending_demand[item.id] += item.qty;
            }
        }

        while !available_aisles.is_empty() {
            if total_items >= data.wave_size_ub { break; }
            
            let mut candidates = Vec::new();
            for &aid in &available_aisles {
                let mut useful = 0;
                for item in &data.dense_aisles[aid] {
                    useful += std::cmp::min(item.qty, pending_demand[item.id]);
                }
                let base_score = useful as f64;
                let score = if let Some(w) = aisle_weights { base_score * w[aid] } else { base_score };
                candidates.push((aid, score));
            }
            
            if candidates.is_empty() { break; }
            
            let (min_s, max_s) = candidates.iter().fold((f64::MAX, f64::MIN), |(min, max), c| (min.min(c.1), max.max(c.1)));
            let threshold = max_s - self.alpha * (max_s - min_s);
            let rcl: Vec<(usize, f64)> = candidates.into_iter().filter(|c| c.1 >= threshold).collect();
            
            if rcl.is_empty() { break; }
            let next_aid = if rcl.len() <= 1 {
                rcl[0].0
            } else {
                let total_weight: f64 = rcl.iter().map(|&(_, s)| s).sum();
                let mut rng_val = rng.random::<f64>() * total_weight;
                let mut chosen = rcl.last().unwrap().0;
                for &(id, s) in &rcl {
                    if rng_val <= s {
                        chosen = id;
                        break;
                    }
                    rng_val -= s;
                }
                chosen
            };

            available_aisles.remove(&next_aid);
            curr_sol.aisles.insert(next_aid);
            
            for item in &data.dense_aisles[next_aid] {
                total_curr_stock[item.id] += item.qty;
            }

            let mut to_add = Vec::new();
            
            let mut candidates_orders = Vec::new();
            for &oid in &unselected {
                let mut can_fulfill = true;
                for item in &data.dense_orders[oid] {
                    if total_curr_stock[item.id] < item.qty { can_fulfill = false; break; }
                }
                if can_fulfill { candidates_orders.push(oid); }
            }
            
            loop {
                candidates_orders.retain(|&oid| {
                    if total_items + data.order_total_items[oid] > data.wave_size_ub { return false; }
                    for item in &data.dense_orders[oid] {
                        if total_curr_stock[item.id] < item.qty { return false; }
                    }
                    true
                });

                if candidates_orders.is_empty() {
                    break;
                }

                let scored_orders: Vec<(usize, f64)> = candidates_orders.iter()
                    .map(|&oid| {
                        let base_score = data.order_total_items[oid] as f64;
                        let score = if let Some(w) = order_weights { base_score * w[oid] } else { base_score };
                        (oid, score)
                    }).collect();

                let (min_s, max_s) = scored_orders.iter()
                    .fold((f64::MAX, f64::MIN), |(min, max), &(_, s)| (min.min(s), max.max(s)));
                
                let threshold = max_s - self.alpha * (max_s - min_s);
                let rcl: Vec<(usize, f64)> = scored_orders.into_iter()
                    .filter(|&(_, s)| s >= threshold)
                    .collect();

                let chosen_oid = if rcl.len() <= 1 {
                    rcl[0].0
                } else {
                    let total_weight: f64 = rcl.iter().map(|&(_, s)| s).sum();
                    let mut rng_val = rng.random::<f64>() * total_weight;
                    let mut chosen = rcl.last().unwrap().0;
                    for &(id, s) in &rcl {
                        if rng_val <= s {
                            chosen = id;
                            break;
                        }
                        rng_val -= s;
                    }
                    chosen
                };

                for item in &data.dense_orders[chosen_oid] {
                    total_curr_stock[item.id] -= item.qty;
                    pending_demand[item.id] = pending_demand[item.id].saturating_sub(item.qty);
                }

                to_add.push(chosen_oid);
                total_items += data.order_total_items[chosen_oid];
                unselected.remove(&chosen_oid);
                candidates_orders.retain(|&oid| oid != chosen_oid);
            }
            
            for oid in to_add {
                curr_sol.orders.insert(oid);
            }
            
            if total_items >= data.wave_size_lb && total_items <= data.wave_size_ub {
                let score = total_items as f64 / curr_sol.aisles.count_ones(..) as f64;
                if score > best_score {
                    best_score = score;
                    best_sol = curr_sol.clone();
                }
            }
        }
        if best_score > 0.0 { best_sol } else { curr_sol }
    }
}
