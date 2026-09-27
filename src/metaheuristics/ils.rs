use crate::solution::{ChallengeSolution, ProblemData};
use crate::heuristics::{SolverStrategy, ConstructiveAlgorithm, LocalSearchAlgorithm, utils};
use crate::evaluator::{StockBalanceEvaluator, Evaluator};
use crate::local_searchs::SearchDimension;
use crate::solver::config::IlsConfig;
use rand::prelude::*;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::time::Instant;

pub struct Ils {
    pub constructive: Box<dyn ConstructiveAlgorithm>,
    pub local_search: Box<dyn LocalSearchAlgorithm>,
    pub dimension: SearchDimension,
    pub config: IlsConfig,
}

impl SolverStrategy for Ils {
    fn solve(&self, data: &ProblemData, seed: u64) -> ChallengeSolution {
        let start_time = Instant::now();
        
        let mut best_overall = ChallengeSolution::new(data.orders.len(), data.aisles.len());
        let mut best_overall_obj = -1.0;

        println!("  {:<8} | {:<12} | {:<10} | {:<9} | {:<10}", "Iter", "Best Obj", "Avg Obj", "LSHit%", "Time(s)");
        println!("  {}", "-".repeat(60));

        // Inicia múltiplos walkers em paralelo (Multi-Start ILS)
        let mut walkers_data: Vec<_> = (0..self.config.walkers).into_par_iter().map(|i| {
            let mut rng = ChaCha8Rng::seed_from_u64(seed + i as u64 + 1000);
            let sol = self.constructive.construct(data, rng.next_u64(), None);
            let obj = utils::compute_objective(&sol, data);
            (rng, sol.clone(), obj, sol, obj, 0) // (rng, curr, curr_obj, best, best_obj, no_imp)
        }).collect();

        // Encontra o melhor inicial
        for walker in &walkers_data {
            if walker.4 > best_overall_obj {
                best_overall_obj = walker.4;
                best_overall = walker.3.clone();
            }
        }

        let mut total_iters = 0;
        
        while start_time.elapsed().as_secs() < self.config.max_time_secs && total_iters < self.config.iterations {
            total_iters += 1;
            
            // Refina todos os walkers em paralelo
            let stats: (f64, usize, usize) = walkers_data.par_iter_mut().map(|walker| {
                let (rng, curr, curr_obj, best, best_obj, no_imp) = walker;
                
                // Perturbação Dinâmica: Aumenta a intensidade se estagnar
                let intensity = if *no_imp > self.config.stagnation_limit { 
                    self.config.perturbation_max 
                } else { 
                    self.config.perturbation_base 
                };

                let mut cand = curr.clone();
                self.perturb(&mut cand, data, intensity, rng);
                
                let refined = self.local_search.refine(&mut cand, data, rng.next_u64());
                let cand_obj = utils::compute_objective(&cand, data);
                
                // Critério de Aceitação (Simples e Agressivo)
                if cand_obj > *best_obj + 1e-6 {
                    // Novo ótimo global para este walker
                    *best = cand.clone();
                    *best_obj = cand_obj;
                    *curr = cand;
                    *curr_obj = cand_obj;
                    *no_imp = 0;
                } else if cand_obj > *curr_obj + 1e-6 {
                    // Melhoria em relação ao passo anterior, mas não global
                    *curr = cand;
                    *curr_obj = cand_obj;
                    *no_imp = 0;
                } else if cand_obj > 0.0 && rng.random_bool(0.1) {
                    // Piora, mas aceita com pequena probabilidade para escapar
                    *curr = cand;
                    *curr_obj = cand_obj;
                    *no_imp += 1;
                    // Rejeita a solução. Se estagnar muito, reinicia a partir da melhor solução (Fallback)
                    *no_imp += 1;
                    if *no_imp > self.config.stagnation_limit * 2 {
                        *curr = best.clone();
                        *curr_obj = *best_obj;
                        *no_imp = 0;
                    }
                }
                
                (*curr_obj, if refined { 1 } else { 0 }, 1)
            }).reduce(|| (0.0, 0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));

            let mut improved_overall = false;
            for walker in &walkers_data {
                if walker.4 > best_overall_obj {
                    best_overall_obj = walker.4;
                    best_overall = walker.3.clone();
                    improved_overall = true;
                }
            }

            if total_iters % self.config.log_frequency == 0 || improved_overall || total_iters == self.config.iterations {
                let avg_obj = stats.0 / self.config.walkers as f64;
                let ls_hit_rate = (stats.1 as f64 / stats.2 as f64) * 100.0;
                println!("  {:<8} | {:<12.4} | {:<10.4} | {:>4.1}%    | {:<10}", 
                    total_iters, best_overall_obj, avg_obj, ls_hit_rate, start_time.elapsed().as_secs());
            }
        }
        
        println!("  {}", "-".repeat(60));
        println!("  [ILS] Finished in {} iterations. Best Obj: {:.4}", total_iters, best_overall_obj);
        best_overall
    }

    fn name(&self) -> String { 
        format!("Parallel ILS [{}w x {}i] on {:?} ({})", self.config.walkers, self.config.iterations, self.dimension, self.local_search.name()) 
    }
}

impl Ils {
    fn perturb(&self, sol: &mut ChallengeSolution, data: &ProblemData, intensity: f64, rng: &mut impl rand::Rng) {
        let dim = self.dimension;
        let n = sol.bits(dim).count_ones(..); 
        if n == 0 { return; }
        
        let nrem = (n as f64 * intensity).ceil() as usize;
        let nrem = nrem.max(2).min(n);
        
        let to_rem: Vec<usize> = sol.bits(dim).ones().choose_multiple(rng, nrem);
        { 
            let bits = sol.bits_mut(dim); 
            for idx in to_rem { 
                bits.remove(idx); 
            } 
        }
        
        let mut eval = StockBalanceEvaluator::new(sol, data);
        eval.sync_dimension(dim, data);
        eval.commit();
        sol.orders = eval.get_active_orders();
        sol.aisles = eval.get_active_aisles();
        
        // Proteção contra destruição excessiva
        if sol.orders.count_ones(..) < (data.wave_size_lb as usize / 2) {
             *sol = self.constructive.construct(data, rng.next_u64(), None);
        }
    }
}
