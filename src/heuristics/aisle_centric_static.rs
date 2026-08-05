use crate::solution::{ChallengeSolution, ProblemData};
use crate::heuristics::ConstructiveAlgorithm;

pub struct AisleCentricStatic;

impl ConstructiveAlgorithm for AisleCentricStatic {
    fn name(&self) -> String { "AisleCentric (Static)".to_string() }

    fn construct(&self, data: &ProblemData, _seed: u64) -> ChallengeSolution {
        let n_orders = data.orders.len();
        let mut curr_sol = ChallengeSolution::new(n_orders, data.aisles.len());
        let mut best_sol = ChallengeSolution::new(n_orders, data.aisles.len());
        let mut best_score = 0.0;
        let mut unselected = vec![true; n_orders];
        let mut curr_stock = vec![0u32; data.aisles.len() * data.n_items];
        let mut total_curr_stock = vec![0u32; data.n_items];
        let mut total_items = 0;

        let mut all_aisles: Vec<usize> = (0..data.aisles.len()).collect();
        all_aisles.sort_by(|&a, &b| {
            let sa: u32 = data.aisles[a].values().sum();
            let sb: u32 = data.aisles[b].values().sum();
            sb.cmp(&sa)
        });


        for &aid in &all_aisles {
            if total_items >= data.wave_size_ub { break; }
            curr_sol.aisles.insert(aid);
            
            for item in &data.dense_aisles[aid] {
                curr_stock[aid * data.n_items + item.id] = item.qty;
                total_curr_stock[item.id] += item.qty;
            }

            let mut to_add = Vec::new();
            for &oid in &data.orders_sorted_by_size {
                if !unselected[oid] { continue; }
                let oqty = data.order_total_items[oid];
                if total_items + oqty > data.wave_size_ub { continue; }

                let mut can = true;
                for item in &data.dense_orders[oid] {
                    if total_curr_stock[item.id] < item.qty {
                        can = false;
                        break;
                    }
                }

                if can {
                    for item in &data.dense_orders[oid] {
                        let mut rem = item.qty;
                        total_curr_stock[item.id] -= item.qty;
                        for aidx in curr_sol.aisles.ones() {
                            if rem == 0 { break; }
                            let s = &mut curr_stock[aidx * data.n_items + item.id];
                            let take = std::cmp::min(rem, *s);
                            *s -= take;
                            rem -= take;
                        }
                    }
                    to_add.push(oid);
                    total_items += oqty;
                }
            }

            for oid in to_add {
                curr_sol.orders.insert(oid);
                unselected[oid] = false;
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
