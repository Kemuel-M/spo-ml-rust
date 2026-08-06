use crate::solution::{ChallengeSolution, ProblemData};
use crate::heuristics::{SolverStrategy, ConstructiveAlgorithm, LocalSearchAlgorithm};
use crate::evaluator::{StockBalanceEvaluator, Evaluator};
use rayon::prelude::*;
use std::time::Instant;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::SeedableRng;

use crate::solver::GraspConfig;

pub struct Grasp {
    pub constructive: Box<dyn ConstructiveAlgorithm>, 
    pub local_search: Box<dyn LocalSearchAlgorithm>,
    pub config: GraspConfig,
}

impl Grasp {
    fn run_iteration(&self, data: &ProblemData, seed: u64, iter_idx: u64, alpha: f64, p_orders: f64) -> (f64, ChallengeSolution) {
        let iter_seed = seed + iter_idx;
        
        let mut solution = if let Some(constr) = self.constructive.with_alpha(alpha) {
            if constr.name().contains("Hybrid") {
                let hybrid = crate::heuristics::hybrid_random::HybridRandom {
                    order_strategy: crate::heuristics::random_greedy::RandomGreedy { alpha },
                    aisle_strategy: crate::heuristics::aisle_centric_random::AisleCentricRandom { alpha },
                    p_orders,
                };
                hybrid.construct(data, iter_seed, None)
            } else {
                constr.construct(data, iter_seed, None)
            }
        } else {
            self.constructive.construct(data, iter_seed, None)
        };

        // Usa um offset para a semente da busca local para não repetir a mesma sequência da construção
        self.local_search.refine(&mut solution, data, iter_seed + 1000);
        let evaluator = StockBalanceEvaluator::new(&solution, data);
        let objective = evaluator.current_objective();
        (objective, solution)
    }
}

impl SolverStrategy for Grasp {
    fn solve(&self, data: &ProblemData, seed: u64) -> ChallengeSolution {
        let start_time = Instant::now();
        
        let mut best_overall_sol = ChallengeSolution::new(data.orders.len(), data.aisles.len());
        let mut best_overall_obj = -1.0;

        let chunk_size = self.config.chunk_size; 
        let mut total_iterations = 0;

        // Probabilidade inicial (0.5)
        let mut p_orders = 0.5;

        println!("  {:<8} | {:<12} | {:<10} | {:<12} | {:<10}", "Iter", "Best Obj", "Avg Chunk", "P_ord/P_ais", "Time");
        println!("  {}", "-".repeat(65));

        while start_time.elapsed().as_secs() < self.config.max_time_secs {
            let chunk_start = Instant::now();
            
            let chunk_results: Vec<(f64, ChallengeSolution)> = (0..chunk_size)
                .into_par_iter()
                .map(|i| {
                    let iter_idx = total_iterations + i as u64;
                    let mut iter_rng = ChaCha8Rng::seed_from_u64(seed + iter_idx);
                    let alpha = iter_rng.random_range(self.config.alpha_min..self.config.alpha_max);
                    self.run_iteration(data, seed, iter_idx, alpha, p_orders)
                })
                .collect();

            let mut chunk_sum_obj = 0.0;
            
            // Stats exclusivas do bloco para reatividade imediata
            let mut chunk_sum_orders = 0.0;
            let mut chunk_count_orders = 0;
            let mut chunk_sum_aisles = 0.0;
            let mut chunk_count_aisles = 0;

            for (obj, sol) in chunk_results {
                chunk_sum_obj += obj;

                if let Some(strat) = sol.metadata.get("strategy") {
                    if strat == "orders" {
                        chunk_sum_orders += obj;
                        chunk_count_orders += 1;
                    } else if strat == "aisles" {
                        chunk_sum_aisles += obj;
                        chunk_count_aisles += 1;
                    }
                }

                if obj > best_overall_obj {
                    best_overall_obj = obj;
                    best_overall_sol = sol;
                }
            }
            
            let chunk_avg = chunk_sum_obj / chunk_size as f64;
            let total_elapsed = start_time.elapsed().as_secs();

            // --- Log a cada bloco ---
            println!("  {:<8} | {:<12.4} | {:<10.4} | {:.2} / {:.2} | {:>4}s ({:.2}s/it)", 
                total_iterations, best_overall_obj, chunk_avg, p_orders, 1.0 - p_orders,
                total_elapsed, chunk_start.elapsed().as_secs_f64() / chunk_size as f64
            );
            
            // --- Lógica Reativa por Bloco (EMA) Robusta ---
            // Só tentamos atualizar se pelo menos uma estratégia foi usada.
            if chunk_count_orders > 0 || chunk_count_aisles > 0 {
                
                // Se uma estratégia não foi testada no bloco, assumimos que o desempenho dela 
                // seria uma penalidade leve (ex: 80% do best_overall_obj) para dar chance de recuperação,
                // em vez de quebrar a fórmula ou causar divisões por zero.
                let avg_orders = if chunk_count_orders > 0 { 
                    chunk_sum_orders / chunk_count_orders as f64 
                } else { 
                    best_overall_obj * self.config.reactive_penalty 
                };

                let avg_aisles = if chunk_count_aisles > 0 { 
                    chunk_sum_aisles / chunk_count_aisles as f64 
                } else { 
                    best_overall_obj * self.config.reactive_penalty 
                };
                
                // Amplificação do desempenho do bloco atual
                let q_orders = (avg_orders / best_overall_obj).powi(self.config.reactive_power);
                let q_aisles = (avg_aisles / best_overall_obj).powi(self.config.reactive_power);
                
                // Evita NaN caso ambas as contas resultem em zero por algum motivo extremo
                if q_orders > 0.0 || q_aisles > 0.0 {
                    let target_p = q_orders / (q_orders + q_aisles);
                    
                    // Média Móvel Exponencial com inércia e limites (clipping)
                    // para garantir que a probabilidade nunca seja exatamente 0 ou 1, 
                    // forçando o algoritmo a testar a outra estratégia eventualmente.
                    let ema = self.config.reactive_ema;
                    p_orders = p_orders * (1.0 - ema) + target_p * ema;
                    p_orders = p_orders.clamp(0.05, 0.95);
                }
            }

            total_iterations += chunk_size as u64;
            if self.config.iterations > 0 && total_iterations >= self.config.iterations as u64 { break; }
        }

        println!("  {}", "-".repeat(65));
        println!("  [GRASP] Finished. Total Iterations: {} | Final Obj: {:.4}", total_iterations, best_overall_obj);
        
        best_overall_sol
    }

    fn name(&self) -> String {
        format!("GRASP [{} iters, {}s] ({}) + ({})", 
            self.config.iterations, self.config.max_time_secs, self.constructive.name(), self.local_search.name()
        )
    }
}
