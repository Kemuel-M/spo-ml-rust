use crate::solution::{ChallengeSolution, ProblemData};
use crate::heuristics::{ConstructiveAlgorithm, utils};
use crate::local_searchs::SearchDimension;
use rand::prelude::*;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

pub struct RandomKeyDecoder {
    pub is_stochastic: bool,
    pub threshold: Option<f64>,
}

impl ConstructiveAlgorithm for RandomKeyDecoder {
    fn name(&self) -> String {
        let mode = if self.is_stochastic { "Stochastic" } else { "Deterministic" };
        if let Some(t) = self.threshold {
            format!("Random Key Decoder ({} w/ Threshold {:.2})", mode, t)
        } else {
            format!("Random Key Decoder ({} Rollback)", mode)
        }
    }

    fn construct(&self, data: &ProblemData, seed: u64, weights: Option<(&[f64], SearchDimension)>) -> ChallengeSolution {
        let (scores, dim) = weights.expect("RandomKeyDecoder requires weights (scores and dimension)");
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let n_orders = data.orders.len();
        
        if dim == SearchDimension::Orders {
            let mut sol = ChallengeSolution::new(n_orders, data.aisles.len());
            let mut available_stock = data.stock_matrix.clone();
            let mut curr_items = 0;
            let mut curr_aisles_count = 0usize;

            let mut order_indices: Vec<usize> = (0..n_orders).collect();
            order_indices.sort_unstable_by(|&a, &b| {
                scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal)
            });

            let mut best_obj = -1.0;
            let mut best_sol: Option<ChallengeSolution> = None;

            for &idx in &order_indices {
                if curr_items >= data.wave_size_ub { break; }
                
                if let Some(thresh) = self.threshold {
                    if scores[idx] < thresh { continue; }
                }

                if self.is_stochastic {
                    if rng.random::<f64>() >= scores[idx] { continue; }
                }

                let total = data.order_total_items[idx];
                if curr_items + total > data.wave_size_ub { continue; }

                if !utils::is_stock_sufficient_dense(&data.dense_orders[idx], &data.order_required_aisles[idx], &available_stock, data.n_items) {
                    continue;
                }

                sol.orders.insert(idx);
                curr_items += total;
                for &a in &data.order_required_aisles[idx] {
                    if !sol.aisles.contains(a) { 
                        sol.aisles.insert(a); 
                        curr_aisles_count += 1;
                    }
                }
                utils::update_stock_dense(&data.dense_orders[idx], &data.item_to_aisles, &mut available_stock, &sol.aisles, data.n_items);

                if self.threshold.is_none() {
                    if curr_items >= data.wave_size_lb {
                        let current_obj = curr_items as f64 / curr_aisles_count.max(1) as f64;
                        if current_obj > best_obj {
                            best_obj = current_obj;
                            let mut snap = sol.clone();
                            snap.score = current_obj;
                            snap.feasible = true;
                            best_sol = Some(snap);
                        }
                    }
                }
            }

            // Fallback (Rede de segurança): garante factibilidade (wave_size_lb) independente do threshold
            if curr_items < data.wave_size_lb {
                for &idx in &order_indices {
                    if curr_items >= data.wave_size_lb { break; } // Atingiu factibilidade, pode parar!
                    if sol.orders.contains(idx) { continue; }
                    
                    let total = data.order_total_items[idx];
                    if curr_items + total > data.wave_size_ub { continue; }

                    if !utils::is_stock_sufficient_dense(&data.dense_orders[idx], &data.order_required_aisles[idx], &available_stock, data.n_items) {
                        continue;
                    }

                    sol.orders.insert(idx);
                    curr_items += total;
                    for &a in &data.order_required_aisles[idx] {
                        if !sol.aisles.contains(a) { 
                            sol.aisles.insert(a); 
                            curr_aisles_count += 1;
                        }
                    }
                    utils::update_stock_dense(&data.dense_orders[idx], &data.item_to_aisles, &mut available_stock, &sol.aisles, data.n_items);
                }
            }

            if self.threshold.is_some() {
                sol.score = curr_items as f64 / curr_aisles_count.max(1) as f64;
                sol.feasible = curr_items >= data.wave_size_lb;
                sol
            } else if let Some(best) = best_sol {
                best
            } else {
                sol.score = curr_items as f64 / curr_aisles_count.max(1) as f64;
                sol.feasible = curr_items >= data.wave_size_lb;
                sol
            }
        } else {
            // Aisle-Centric
            let mut sol = ChallengeSolution::new(n_orders, data.aisles.len());
            let mut curr_stock = vec![0u32; data.aisles.len() * data.n_items];
            let mut total_curr_stock = vec![0u32; data.n_items];
            let mut total_items = 0;
            let mut curr_aisles_count = 0usize;
            
            let mut aisle_indices: Vec<usize> = (0..data.aisles.len()).collect();
            aisle_indices.sort_unstable_by(|&a, &b| {
                scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal)
            });

            let order_indices = &data.orders_sorted_by_size;
            let mut best_obj = -1.0;
            let mut best_sol: Option<ChallengeSolution> = None;

            for &aid in &aisle_indices {
                if total_items >= data.wave_size_ub { break; }
                
                if let Some(thresh) = self.threshold {
                    if scores[aid] < thresh { continue; }
                }

                if self.is_stochastic {
                    if rng.random::<f64>() >= scores[aid] { continue; }
                }

                sol.aisles.insert(aid);
                curr_aisles_count += 1;

                for item in &data.dense_aisles[aid] {
                    curr_stock[aid * data.n_items + item.id] = item.qty;
                    total_curr_stock[item.id] += item.qty;
                }

                for &oid in order_indices {
                    if sol.orders.contains(oid) { continue; }
                    let oqty = data.order_total_items[oid];
                    if total_items + oqty > data.wave_size_ub { continue; }

                    let mut can_fulfill = true;
                    for item in &data.dense_orders[oid] {
                        if total_curr_stock[item.id] < item.qty { can_fulfill = false; break; }
                    }

                    if can_fulfill {
                        for item in &data.dense_orders[oid] {
                            total_curr_stock[item.id] -= item.qty;
                        }
                        sol.orders.insert(oid);
                        total_items += oqty;
                    }
                }

                if self.threshold.is_none() {
                    if total_items >= data.wave_size_lb {
                        let current_obj = total_items as f64 / curr_aisles_count.max(1) as f64;
                        if current_obj > best_obj {
                            best_obj = current_obj;
                            let mut snap = sol.clone();
                            snap.score = current_obj;
                            snap.feasible = true;
                            best_sol = Some(snap);
                        }
                    }
                }
            }
            
            // Fallback (Rede de segurança): garante factibilidade (wave_size_lb) independente do threshold
            if total_items < data.wave_size_lb {
                for &aid in &aisle_indices {
                    if total_items >= data.wave_size_lb { break; } // Atingiu factibilidade, pode parar!
                    if sol.aisles.contains(aid) { continue; }
                    
                    sol.aisles.insert(aid);
                    curr_aisles_count += 1;

                    for item in &data.dense_aisles[aid] {
                        curr_stock[aid * data.n_items + item.id] = item.qty;
                        total_curr_stock[item.id] += item.qty;
                    }

                    for &oid in order_indices {
                        if sol.orders.contains(oid) { continue; }
                        let oqty = data.order_total_items[oid];
                        if total_items + oqty > data.wave_size_ub { continue; }

                        let mut can_fulfill = true;
                        for item in &data.dense_orders[oid] {
                            if total_curr_stock[item.id] < item.qty { can_fulfill = false; break; }
                        }

                        if can_fulfill {
                            for item in &data.dense_orders[oid] {
                                total_curr_stock[item.id] -= item.qty;
                            }
                            sol.orders.insert(oid);
                            total_items += oqty;
                        }
                    }
                }
            }

            if self.threshold.is_some() {
                sol.score = total_items as f64 / curr_aisles_count.max(1) as f64;
                sol.feasible = total_items >= data.wave_size_lb && total_items <= data.wave_size_ub;
                sol
            } else if let Some(best) = best_sol {
                best
            } else {
                sol.score = total_items as f64 / curr_aisles_count.max(1) as f64;
                sol.feasible = total_items >= data.wave_size_lb && total_items <= data.wave_size_ub;
                sol
            }
        }
    }
}
