use crate::solution::{ChallengeSolution, ProblemData};
use crate::heuristics::{SolverStrategy, ConstructiveAlgorithm, LocalSearchAlgorithm, utils};
use crate::evaluator::{StockBalanceEvaluator, Evaluator};
use crate::local_searchs::SearchDimension;
use rand::prelude::*;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

pub struct Ils {
    pub constructive: Box<dyn ConstructiveAlgorithm>,
    pub local_search: Box<dyn LocalSearchAlgorithm>,
    pub dimension: SearchDimension,
    pub max_iterations: usize,
    pub max_time_secs: u64,
}

impl SolverStrategy for Ils {
    fn solve(&self, data: &ProblemData, seed: u64) -> ChallengeSolution {
        let start_time = Instant::now();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut curr = self.constructive.construct(data, seed, None);
        self.local_search.refine(&mut curr, data, seed + 1);
        let mut best = curr.clone();
        let mut best_obj = utils::compute_objective(&best, data);
        let mut no_imp = 0;

        for _ in 0..self.max_iterations {
            if start_time.elapsed().as_secs() >= self.max_time_secs {
                break;
            }

            let intensity = if no_imp > 5 { 0.30 } else { 0.15 };
            let mut cand = curr.clone(); 
            self.perturb(&mut cand, data, intensity, &mut rng);
            self.local_search.refine(&mut cand, data, rng.next_u64());
            let cand_obj = utils::compute_objective(&cand, data);
            if cand_obj > best_obj + 1e-6 {
                best = cand.clone(); best_obj = cand_obj; curr = cand; no_imp = 0;
            } else if cand_obj > 0.0 && rng.random_bool(0.1) {
                curr = cand; no_imp += 1;
            } else {
                curr = best.clone(); no_imp += 1;
            }
        }
        best
    }
    fn name(&self) -> String { format!("ILS [{} iters, {}s] on {:?} ({})", self.max_iterations, self.max_time_secs, self.dimension, self.local_search.name()) }
}

impl Ils {
    fn perturb(&self, sol: &mut ChallengeSolution, data: &ProblemData, intensity: f64, rng: &mut impl rand::Rng) {
        let dim = self.dimension;
        let n = sol.bits(dim).count_ones(..); if n == 0 { return; }
        let nrem = (n as f64 * intensity).ceil() as usize;
        let nrem = nrem.max(2).min(n);
        let to_rem: Vec<usize> = sol.bits(dim).ones().choose_multiple(rng, nrem);
        { let bits = sol.bits_mut(dim); for idx in to_rem { bits.remove(idx); } }
        let mut eval = StockBalanceEvaluator::new(sol, data);
        eval.sync_dimension(dim, data);
        eval.commit();
        sol.orders = eval.get_active_orders();
        sol.aisles = eval.get_active_aisles();
        if sol.orders.count_ones(..) < (data.wave_size_lb as usize / 2) {
             *sol = self.constructive.construct(data, rng.next_u64(), None);
        }
    }
}
