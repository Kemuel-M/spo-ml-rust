use crate::solution::{ChallengeSolution, ProblemData};
use crate::heuristics::ConstructiveAlgorithm;
use crate::heuristics::random_greedy::RandomGreedy;
use crate::heuristics::aisle_centric_random::AisleCentricRandom;
use rand::prelude::*;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

pub enum StrategyChoice {
    Orders,
    Aisles,
}

pub struct HybridRandom {
    pub order_strategy: RandomGreedy,
    pub aisle_strategy: AisleCentricRandom,
    pub p_orders: f64, // Probabilidade de escolher Pedidos (0.0 a 1.0)
}

impl HybridRandom {
    pub fn new(alpha: f64) -> Self {
        Self {
            order_strategy: RandomGreedy { alpha },
            aisle_strategy: AisleCentricRandom { alpha },
            p_orders: 0.5,
        }
    }
}

impl ConstructiveAlgorithm for HybridRandom {
    fn name(&self) -> String {
        format!("Hybrid (P_ord={:.2}, P_ais={:.2})", self.p_orders, 1.0 - self.p_orders)
    }

    fn with_alpha(&self, alpha: f64) -> Option<Box<dyn ConstructiveAlgorithm>> {
        Some(Box::new(HybridRandom {
            order_strategy: RandomGreedy { alpha },
            aisle_strategy: AisleCentricRandom { alpha },
            p_orders: self.p_orders,
        }))
    }

    fn construct(&self, data: &ProblemData, seed: u64) -> ChallengeSolution {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        
        let sol = if rng.random_bool(self.p_orders) {
            let mut s = self.order_strategy.construct(data, seed);
            s.metadata.insert("strategy".to_string(), "orders".to_string());
            s
        } else {
            let mut s = self.aisle_strategy.construct(data, seed);
            s.metadata.insert("strategy".to_string(), "aisles".to_string());
            s
        };
        sol
    }
}
