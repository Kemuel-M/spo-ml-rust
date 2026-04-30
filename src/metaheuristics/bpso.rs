use crate::solution::{ChallengeSolution, ProblemData};
use crate::heuristics::{SolverStrategy, ConstructiveAlgorithm, LocalSearchAlgorithm, utils};
use crate::local_searchs::SearchDimension;
use crate::evaluator::{Evaluator, StockBalanceEvaluator};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::time::{Instant, Duration};

pub struct BPSOConfig {
    pub dimension: SearchDimension, pub population_size: usize, pub iterations: usize,
    pub w_min: f64, pub w_max: f64, pub c1: f64, pub c2: f64, pub v_max: f64, pub ls_prob: f64, pub max_time_secs: u64,
}

pub struct BPSO {
    pub constructive: Box<dyn ConstructiveAlgorithm>, pub local_search: Box<dyn LocalSearchAlgorithm>, pub config: BPSOConfig,
}

struct Particle {
    position: Vec<bool>, velocity: Vec<f64>, best_position: Vec<bool>,
    best_fitness: f64, best_solution: ChallengeSolution, current_solution: ChallengeSolution,
}

impl BPSO {
    fn sigmoid(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }
    fn solution_to_vec(&self, sol: &ChallengeSolution, n: usize) -> Vec<bool> {
        let mut v = vec![false; n]; let bits = sol.bits(self.config.dimension);
        for idx in bits.ones() { if idx < n { v[idx] = true; } } v
    }
    fn vec_to_solution(&self, v: &[bool], no: usize, na: usize) -> ChallengeSolution {
        let mut sol = ChallengeSolution::new(no, na); let bits = sol.bits_mut(self.config.dimension);
        for (i, &s) in v.iter().enumerate() { if s { bits.insert(i); } } sol
    }
    fn repair(&self, sol: &mut ChallengeSolution, data: &ProblemData, seed: u64) {
        let mut eval = StockBalanceEvaluator::new(sol, data);
        eval.sync_dimension(self.config.dimension, data); eval.commit();
        sol.orders = eval.get_active_orders(); sol.aisles = eval.get_active_aisles();
        if !utils::is_feasible(sol, data) { *sol = self.constructive.construct(data, seed); }
    }
}

impl SolverStrategy for BPSO {
    fn name(&self) -> String { format!("BPSO on {:?} [{} + {}]", self.config.dimension, self.constructive.name(), self.local_search.name()) }
    fn solve(&self, data: &ProblemData, seed: u64) -> ChallengeSolution {
        let start = Instant::now(); let max_dur = Duration::from_secs(self.config.max_time_secs);
        let n = match self.config.dimension { SearchDimension::Orders => data.orders.len(), SearchDimension::Aisles => data.aisles.len() };
        let mut particles: Vec<Particle> = (0..self.config.population_size).into_par_iter().map(|i| {
            let s = self.constructive.construct(data, seed + i as u64); let f = utils::compute_objective(&s, data);
            let p = self.solution_to_vec(&s, n); let mut v = vec![0.0; n];
            let mut rng = ChaCha8Rng::seed_from_u64(seed + i as u64 + 1000);
            for val in v.iter_mut() { *val = rng.random_range(-self.config.v_max..self.config.v_max); }
            Particle { position: p.clone(), velocity: v, best_position: p, best_fitness: f, best_solution: s.clone(), current_solution: s }
        }).collect();

        let mut gbest_f = -1.0; let mut gbest_idx = 0;
        for (i, p) in particles.iter().enumerate() { if p.best_fitness > gbest_f { gbest_f = p.best_fitness; gbest_idx = i; } }
        let mut gbest_p = particles[gbest_idx].best_position.clone();
        let mut gbest_s = particles[gbest_idx].best_solution.clone();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        for iter in 0..self.config.iterations {
            if start.elapsed() >= max_dur { break; }
            let w = self.config.w_max - (self.config.w_max - self.config.w_min) * (iter as f64 / self.config.iterations as f64);
            let base = rng.next_u64();
            particles.par_iter_mut().enumerate().for_each(|(i, p)| {
                let mut lrng = ChaCha8Rng::seed_from_u64(base + i as u64);
                for j in 0..n {
                    let r1 = lrng.random::<f64>(); let r2 = lrng.random::<f64>();
                    let cog = self.config.c1 * r1 * (if p.best_position[j] { 1.0 } else { 0.0 } - if p.position[j] { 1.0 } else { 0.0 });
                    let soc = self.config.c2 * r2 * (if gbest_p[j] { 1.0 } else { 0.0 } - if p.position[j] { 1.0 } else { 0.0 });
                    p.velocity[j] = w * p.velocity[j] + cog + soc;
                    if p.velocity[j] > self.config.v_max { p.velocity[j] = self.config.v_max; }
                    if p.velocity[j] < -self.config.v_max { p.velocity[j] = -self.config.v_max; }
                    p.position[j] = lrng.random::<f64>() < Self::sigmoid(p.velocity[j]);
                }
                let mut sol = self.vec_to_solution(&p.position, data.orders.len(), data.aisles.len());
                self.repair(&mut sol, data, base + i as u64 + 5000);
                if lrng.random_bool(self.config.ls_prob) { self.local_search.refine(&mut sol, data, base + i as u64 + 10000); }
                let f = utils::compute_objective(&sol, data);
                p.current_solution = sol; p.position = self.solution_to_vec(&p.current_solution, n);
                if f > p.best_fitness { p.best_fitness = f; p.best_position = p.position.clone(); p.best_solution = p.current_solution.clone(); }
            });
            for p in &particles { if p.best_fitness > gbest_f { gbest_f = p.best_fitness; gbest_p = p.best_position.clone(); gbest_s = p.best_solution.clone(); } }
        }
        gbest_s
    }
}
