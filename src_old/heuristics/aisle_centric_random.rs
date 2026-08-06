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

    fn construct(&self, data: &ProblemData, seed: u64) -> ChallengeSolution {
        let n_orders = data.orders.len();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut curr_sol = ChallengeSolution::new(n_orders, data.aisles.len());
        let mut best_sol = ChallengeSolution::new(n_orders, data.aisles.len());
        let mut best_score = 0.0;
        let mut unselected: HashSet<usize> = (0..n_orders).collect();
        let mut available_aisles: HashSet<usize> = (0..data.aisles.len()).collect();
        
        let mut curr_stock = vec![0u32; data.aisles.len() * data.n_items];
        let mut total_curr_stock = vec![0u32; data.n_items];
        let mut total_items = 0;

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
                candidates.push((aid, useful as f64));
            }
            
            if candidates.is_empty() { break; }
            
            let (min_s, max_s) = candidates.iter().fold((f64::MAX, f64::MIN), |(min, max), c| (min.min(c.1), max.max(c.1)));
            let threshold = max_s - self.alpha * (max_s - min_s);
            let rcl: Vec<usize> = candidates.into_iter().filter(|c| c.1 >= threshold).map(|c| c.0).collect();
            
            if rcl.is_empty() { break; }
            let next_aid = *rcl.choose(&mut rng).unwrap();

            available_aisles.remove(&next_aid);
            curr_sol.aisles.insert(next_aid);
            
            for item in &data.dense_aisles[next_aid] {
                curr_stock[next_aid * data.n_items + item.id] = item.qty;
                total_curr_stock[item.id] += item.qty;
            }

            let mut to_add = Vec::new();
            
            // Em vez de percorrer uma lista estática, preenchemos a onda iterativamente 
            // usando uma abordagem GRASP para os pedidos também.
            loop {
                let mut candidates_orders = Vec::new();

                for &oid in &unselected {
                    let oqty = data.order_total_items[oid];
                    if total_items + oqty > data.wave_size_ub { continue; }

                    let mut can_fulfill = true;
                    for item in &data.dense_orders[oid] {
                        if total_curr_stock[item.id] < item.qty {
                            can_fulfill = false;
                            break;
                        }
                    }

                    if can_fulfill {
                        // A métrica gulosa para o pedido: Tamanho do pedido (favorecemos pedidos maiores para encher a onda)
                        candidates_orders.push((oid, oqty as f64));
                    }
                }

                if candidates_orders.is_empty() {
                    break; // Nenhum pedido restante consegue ser montado com o estoque atual dos corredores abertos
                }

                // Aplica a RCL (Restricted Candidate List) nos pedidos
                let (min_s, max_s) = candidates_orders.iter()
                    .fold((f64::MAX, f64::MIN), |(min, max), c| (min.min(c.1), max.max(c.1)));
                
                let threshold = max_s - self.alpha * (max_s - min_s);
                let rcl: Vec<usize> = candidates_orders.into_iter()
                    .filter(|c| c.1 >= threshold)
                    .map(|c| c.0)
                    .collect();

                let chosen_oid = *rcl.choose(&mut rng).unwrap();

                // Deduz o estoque do pedido escolhido
                for item in &data.dense_orders[chosen_oid] {
                    let mut rem = item.qty;
                    total_curr_stock[item.id] -= item.qty;
                    for aidx in curr_sol.aisles.ones() {
                        if rem == 0 { break; }
                        let s = &mut curr_stock[aidx * data.n_items + item.id];
                        let take = std::cmp::min(rem, *s);
                        *s -= take;
                        rem -= take;
                    }
                    pending_demand[item.id] = pending_demand[item.id].saturating_sub(item.qty);
                }

                to_add.push(chosen_oid);
                total_items += data.order_total_items[chosen_oid];
                unselected.remove(&chosen_oid);
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
