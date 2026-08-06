use crate::solution::{ChallengeSolution, ProblemData};
use crate::heuristics::ConstructiveAlgorithm;
use crate::heuristics::random_greedy::RandomGreedy;
use crate::heuristics::aisle_centric_random::AisleCentricRandom;
use rand::prelude::*;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

pub struct SuperHybrid {
    pub p_zero_alpha: f64,
    pub p_orders: f64,
}

impl SuperHybrid {
    pub fn new() -> Self {
        Self {
            p_zero_alpha: 0.10, // 10% de chance de alpha = 0.0
            p_orders: 0.40,     // 40% de chance de focar em Pedidos (60% para Corredores)
        }
    }
}

impl ConstructiveAlgorithm for SuperHybrid {
    fn name(&self) -> String {
        format!("SuperHybrid (P_zero={:.2}, P_ord={:.2})", self.p_zero_alpha, self.p_orders)
    }

    fn construct(&self, data: &ProblemData, seed: u64) -> ChallengeSolution {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        
        // 1. Decide o Alpha
        let alpha = if rng.random_bool(self.p_zero_alpha) {
            0.0
        } else {
            rng.random_range(0.05..=0.30)
        };
        
        // 2. Decide a Família (Order ou Aisle)
        let is_order = rng.random_bool(self.p_orders);
        
        let sol = if is_order {
            let strat = RandomGreedy { alpha };
            let mut s = strat.construct(data, seed);
            s.metadata.insert("strategy".to_string(), "orders".to_string());
            s.metadata.insert("alpha".to_string(), alpha.to_string());
            s
        } else {
            let strat = AisleCentricRandom { alpha };
            let mut s = strat.construct(data, seed);
            s.metadata.insert("strategy".to_string(), "aisles".to_string());
            s.metadata.insert("alpha".to_string(), alpha.to_string());
            s
        };
        
        sol
    }
}
