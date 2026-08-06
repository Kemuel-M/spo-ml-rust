use crate::solution::{ChallengeSolution, ProblemData};
use crate::heuristics::{SolverStrategy, ConstructiveAlgorithm, LocalSearchAlgorithm, utils};
use crate::local_searchs::SearchDimension;
use crate::evaluator::{Evaluator, StockBalanceEvaluator};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::time::{Instant, Duration};

pub struct GAConfig {
    pub dimension: SearchDimension,
    pub population_size: usize,
    pub generations: usize,
    pub mutation_rate: f64,
    pub elitism_count: usize,
    pub max_time_secs: u64,
    pub memetic_prob: f64,
    pub repair_search_limit: usize,
    pub seed: u64,
}

pub struct GeneticAlgorithm {
    pub constructive: Box<dyn ConstructiveAlgorithm>,
    pub local_search: Box<dyn LocalSearchAlgorithm>,
    pub config: GAConfig,
}

impl SolverStrategy for GeneticAlgorithm {
    fn name(&self) -> String { format!("GA on {:?} [{} + {}]", self.config.dimension, self.constructive.name(), self.local_search.name()) }

    fn solve(&self, data: &ProblemData, seed: u64) -> ChallengeSolution {
        let start = Instant::now();
        let total_budget = Duration::from_secs(self.config.max_time_secs);
        let ga_budget = total_budget.mul_f64(0.85); // 85% para evolução
        let _ls_budget = total_budget.mul_f64(0.15); // 15% para polimento final

        // 1. Inicialização Paralela (Prioridade 4)
        let mut pop: Vec<ChallengeSolution> = (0..self.config.population_size)
            .into_par_iter()
            .map(|i| self.constructive.construct(data, seed + i as u64, None))
            .collect();

        // Aplicar LS inicial em paralelo
        pop.par_iter_mut().enumerate().for_each(|(i, ind)| {
            if ind.feasible {
                let f_before = ind.score;
                let b = ind.clone();
                self.local_search.refine(ind, data, seed + 1000 + i as u64);
                
                // Atualiza score cacheado após busca local
                let eval = StockBalanceEvaluator::new(ind, data);
                ind.score = eval.current_objective();
                ind.feasible = eval.is_stock_valid();
                if ind.score < f_before { *ind = b; }
            }
        });

        let mut best_all = pop[0].clone();
        for p in &pop { if p.score > best_all.score { best_all = p.clone(); } }
        
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        // 2. Loop de Evolução Controlado por Tempo (Prioridade 3)
        let mut generation_count = 0;
        while start.elapsed() < ga_budget && generation_count < self.config.generations {
            generation_count += 1;
            
            // Fits usa o cache O(1) da solução (Prioridade 1)
            let fits: Vec<(usize, f64)> = pop.iter().enumerate().map(|(i, ind)| (i, ind.score)).collect();
            let mut sorted: Vec<usize> = (0..pop.len()).collect();
            sorted.sort_by(|&a, &b| fits[b].1.partial_cmp(&fits[a].1).unwrap_or(std::cmp::Ordering::Equal));
            
            if fits[sorted[0]].1 > best_all.score { 
                best_all = pop[sorted[0]].clone(); 
            }

            let mut next = Vec::with_capacity(self.config.population_size);
            for i in 0..self.config.elitism_count { next.push(pop[sorted[i]].clone()); }

            while next.len() < self.config.population_size {
                let p1 = self.tournament(&pop, &fits, 3, &mut rng);
                let p2 = self.tournament(&pop, &fits, 3, &mut rng);
                let mut offspring = self.crossover(p1, p2, &mut rng);
                self.mutate(&mut offspring, self.config.mutation_rate, &mut rng, data);
                self.repair(&mut offspring, data, rng.next_u64());
                next.push(offspring);
            }

            // Memetismo dinâmico: só aplica se houver tempo
            let base_seed = rng.next_u64();
            let can_do_ls = start.elapsed() < ga_budget.mul_f64(0.9); 

            next.par_iter_mut().enumerate().for_each(|(i, ind)| {
                if i >= self.config.elitism_count && can_do_ls {
                    let mut lrng = ChaCha8Rng::seed_from_u64(base_seed + i as u64);
                    if lrng.random_bool(self.config.memetic_prob) {
                        let f_before = ind.score;
                        let b = ind.clone();
                        self.local_search.refine(ind, data, base_seed + i as u64);
                        
                        let eval = StockBalanceEvaluator::new(ind, data);
                        ind.score = eval.current_objective();
                        ind.feasible = eval.is_stock_valid();
                        if ind.score < f_before { *ind = b; }
                    }
                }
            });
            pop = next;
        }

        // 3. Polimento Final (Prioridade 3)
        // Usa o tempo restante para uma busca local exaustiva no melhor de todos
        if start.elapsed() < total_budget {
            let mut final_best = best_all.clone();
            self.local_search.refine(&mut final_best, data, seed + 99999);
            let eval = StockBalanceEvaluator::new(&final_best, data);
            final_best.score = eval.current_objective();
            if final_best.score > best_all.score {
                best_all = final_best;
            }
        }

        best_all
    }
}

impl GeneticAlgorithm {
    fn tournament<'a>(&self, pop: &'a [ChallengeSolution], fits: &[(usize, f64)], size: usize, rng: &mut impl Rng) -> &'a ChallengeSolution {
        let mut best = &fits[0];
        for _ in 0..size {
            let c = &fits[rng.random_range(0..fits.len())];
            if c.1 > best.1 { best = c; }
        }
        &pop[best.0]
    }
    fn crossover(&self, p1: &ChallengeSolution, p2: &ChallengeSolution, rng: &mut impl Rng) -> ChallengeSolution {
        let mut c = ChallengeSolution::new(p1.orders.len(), p1.aisles.len());
        let d = self.config.dimension;
        let bits = c.bits_mut(d);
        for idx in p1.bits(d).ones() { if rng.random_bool(0.5) { bits.insert(idx); } }
        for idx in p2.bits(d).ones() { if !bits.contains(idx) && rng.random_bool(0.5) { bits.insert(idx); } }
        c
    }
    fn mutate(&self, sol: &mut ChallengeSolution, rate: f64, rng: &mut impl Rng, data: &ProblemData) {
        let d = self.config.dimension;
        let n = match d { SearchDimension::Orders => data.orders.len(), SearchDimension::Aisles => data.aisles.len() };
        let muts = (n as f64 * rate).max(1.0) as usize;
        let bits = sol.bits_mut(d);
        for _ in 0..muts { let i = rng.random_range(0..n); let v = bits.contains(i); bits.set(i, !v); }
    }
    fn repair(&self, sol: &mut ChallengeSolution, data: &ProblemData, seed: u64) {
        let mut eval = StockBalanceEvaluator::new(sol, data);
        eval.sync_dimension(self.config.dimension, data);
        eval.commit();
        sol.orders = eval.get_active_orders();
        sol.aisles = eval.get_active_aisles();
        if !utils::is_feasible(sol, data) { *sol = self.constructive.construct(data, seed, None); }
    }
}
