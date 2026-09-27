use crate::solution::{ChallengeSolution, ProblemData};
use crate::heuristics::{SolverStrategy, ConstructiveAlgorithm, LocalSearchAlgorithm};
use crate::evaluator::{StockBalanceEvaluator, Evaluator};
use rand::prelude::*;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use crate::local_searchs::SearchDimension;
use crate::solver::config::SaConfig;

pub struct SimulatedAnnealing {
    pub constructive: Box<dyn ConstructiveAlgorithm>,
    pub local_search: Box<dyn LocalSearchAlgorithm>,
    pub dimension: SearchDimension,
    pub config: SaConfig,
}

impl SolverStrategy for SimulatedAnnealing {
    fn name(&self) -> String {
        format!("SA [alpha={:.4}] on {:?} (LS: {})", self.config.cooling, self.dimension, self.local_search.name())
    }

    fn solve(&self, data: &ProblemData, seed: u64) -> ChallengeSolution {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut curr_sol = self.constructive.construct(data, seed, None);
        
        let mut best_sol = curr_sol.clone();
        let mut eval = StockBalanceEvaluator::new(&curr_sol, data);
        let mut curr_obj = eval.current_objective();
        let mut best_obj = curr_obj;
        
        let orders_vec: Vec<usize> = (0..data.orders.len()).collect();
        let aisles_vec: Vec<usize> = (0..data.aisles.len()).collect();

        // Calcula T0 inicial com semente independente
        let mut rng_t0 = ChaCha8Rng::seed_from_u64(seed + 9999);
        let mut initial_temp = self.calculate_initial_temp(data, &curr_sol, curr_obj, &orders_vec, &aisles_vec, &mut rng_t0);
        
        // Garante temperatura mínima proporcional ao score inicial
        initial_temp = initial_temp.max(curr_obj * self.config.survival_fallback_ratio);
        let mut temp = initial_temp;

        // Iterações por patamar de temperatura baseadas no tamanho do problema
        let n = match self.dimension {
            SearchDimension::Orders => data.orders.len(),
            SearchDimension::Aisles => data.aisles.len(),
        };
        let iter_per_temp = (if n > 1000 { n } else { 2 * n }).max(self.config.iter_per_temp);

        let start = std::time::Instant::now();
        let mut iterations_since_improvement = 0;
        let mut reheat_count = 0;
        let stagnation_threshold = (n / self.config.stagnation_threshold_divisor).max(self.config.stagnation_threshold_min); 
        let max_reheats = self.config.max_reheats; 
        let mut temp_steps = 0;
        let mut total_moves = 0;

        while temp > self.config.min_temp {
            if start.elapsed().as_secs() >= self.config.max_time_secs { 
                println!("  [SA] Timeout reached after {} temp steps", temp_steps);
                break; 
            }
            
            let mut improved_this_temp = false;

            // Probabilidades dinâmicas de vizinhança ao longo do resfriamento
            let progress = (temp / initial_temp).min(1.0);
            
            let swap_prob = (self.config.swap_prob_end - (self.config.swap_prob_end - self.config.swap_prob_start) * progress) as u32; 
            let block_prob = (self.config.block_prob_end + (self.config.block_prob_start - self.config.block_prob_end) * progress) as u32; 
            let ins_prob = (self.config.ins_prob_end + (self.config.ins_prob_start - self.config.ins_prob_end) * progress) as u32;   

            for _ in 0..iter_per_temp {
                total_moves += 1;
                let mtype = rng.random_range(0..100);
                let applied = match self.dimension {
                    SearchDimension::Orders => {
                        if mtype < swap_prob {
                            if let Some((r, a)) = self.get_swap(&curr_sol, &orders_vec, &mut rng) { eval.try_apply_order_swap(r, a, data); true } else { false }
                        } else if mtype < swap_prob + block_prob {
                            if let Some((rs, ads)) = self.get_block_swap(&curr_sol, &orders_vec, data, &mut rng) {
                                for &r in &rs { eval.try_apply_order_remove(r, data); }
                                for &a in &ads { eval.try_apply_order_add(a, data); }
                                true
                            } else { false }
                        } else if mtype < swap_prob + block_prob + ins_prob {
                            if let Some(a) = self.get_ins(&curr_sol, &orders_vec, data, &mut rng) { eval.try_apply_order_add(a, data); true } else { false }
                        } else {
                            if let Some(r) = self.get_rem(&curr_sol, data, &mut rng) { eval.try_apply_order_remove(r, data); true } else { false }
                        }
                    },
                    SearchDimension::Aisles => {
                        if mtype < swap_prob {
                            if let Some((r, a)) = self.get_a_swap(&curr_sol, &aisles_vec, &mut rng) { eval.try_apply_aisle_swap(r, a, data); true } else { false }
                        } else if mtype < swap_prob + ins_prob {
                            if let Some(a) = self.get_a_ins(&curr_sol, &aisles_vec, &mut rng) { eval.try_apply_aisle_add(a, data); true } else { false }
                        } else {
                            if let Some(r) = self.get_a_rem(&curr_sol, &mut rng) { eval.try_apply_aisle_remove(r, data); true } else { false }
                        }
                    }
                };

                if applied {
                    let nobj = eval.current_objective();
                    let delta = nobj - curr_obj;
                    
                    if delta >= 0.0 || rng.random::<f64>() < (delta / temp).exp() {
                        eval.commit();
                        curr_sol.orders = eval.get_active_orders();
                        curr_sol.aisles = eval.get_active_aisles();
                        curr_obj = nobj;
                        
                        if curr_obj > best_obj { 
                            best_sol = curr_sol.clone(); 
                            best_obj = curr_obj; 
                            improved_this_temp = true;
                            reheat_count = 0; 
                        }
                    } else { eval.rollback(); }
                }
            }

            temp_steps += 1;
            if improved_this_temp {
                iterations_since_improvement = 0;
            } else {
                iterations_since_improvement += 1;
            }

            if temp_steps % self.config.log_frequency == 0 {
                println!("  [SA] Step: {:>5} | Temp: {:>10.4} | Best: {:>10.4} | Time: {:>4}s", 
                    temp_steps, temp, best_obj, start.elapsed().as_secs());
            }

            // Reaquecimento adaptativo e perturbação ao detectar estagnação
            if iterations_since_improvement > stagnation_threshold && reheat_count < max_reheats {
                reheat_count += 1;
                let reheat_factor = (self.config.reheat_factor_base + self.config.reheat_factor_step * (reheat_count as f64)).min(self.config.reheat_factor_max);

                self.local_search.refine(&mut best_sol, data, rng.next_u64());
                let refined_eval = StockBalanceEvaluator::new(&best_sol, data);
                best_obj = refined_eval.current_objective();

                temp = initial_temp * reheat_factor; 
                curr_sol = best_sol.clone();
                
                let mut kick_eval = StockBalanceEvaluator::new(&curr_sol, data);
                let mut active: Vec<usize> = curr_sol.orders.ones().collect();
                active.shuffle(&mut rng); 
                
                let kick_strength = (active.len() / self.config.kick_strength_divisor).max(self.config.kick_min_orders).min(self.config.kick_max_orders);
                
                for &order in active.iter().take(kick_strength) {
                    kick_eval.try_apply_order_remove(order, data);
                    kick_eval.commit(); 
                }
                
                curr_sol.orders = kick_eval.get_active_orders();
                curr_sol.aisles = kick_eval.get_active_aisles();
                
                eval = StockBalanceEvaluator::new(&curr_sol, data);
                curr_obj = eval.current_objective();
                
                iterations_since_improvement = 0;
                println!("  [SA] Step {}: Reheat #{} to {:.0}% of T0 ({:.4}) + Kick Applied", 
                    temp_steps, reheat_count, reheat_factor * 100.0, temp);
            }            
            temp *= self.config.cooling;
        }
        
        self.local_search.refine(&mut best_sol, data, rng.next_u64());
        println!("  [SA] Finished after {} moves and {} temp steps. Final Obj: {:.4}", total_moves, temp_steps, best_obj);
        best_sol
    }
}

impl SimulatedAnnealing {
    fn calculate_initial_temp(&self, data: &ProblemData, curr_sol: &ChallengeSolution, curr_obj: f64, orders_vec: &[usize], aisles_vec: &[usize], rng: &mut impl Rng) -> f64 {
        let mut deltas = Vec::new();
        let mut eval = StockBalanceEvaluator::new(curr_sol, data);
        
        for _ in 0..self.config.initial_temp_samples {
            let mtype = rng.random_range(0..100);
            let applied = match self.dimension {
                SearchDimension::Orders => {
                    if mtype < self.config.init_temp_swap_prob {
                        if let Some((r, a)) = self.get_swap(curr_sol, orders_vec, rng) { eval.try_apply_order_swap(r, a, data); true } else { false }
                    } else if mtype < self.config.init_temp_swap_prob + self.config.init_temp_ins_prob {
                        if let Some(a) = self.get_ins(curr_sol, orders_vec, data, rng) { eval.try_apply_order_add(a, data); true } else { false }
                    } else {
                        if let Some(r) = self.get_rem(curr_sol, data, rng) { eval.try_apply_order_remove(r, data); true } else { false }
                    }
                },
                SearchDimension::Aisles => {
                    if mtype < self.config.init_temp_swap_prob {
                        if let Some((r, a)) = self.get_a_swap(curr_sol, aisles_vec, rng) { eval.try_apply_aisle_swap(r, a, data); true } else { false }
                    } else if mtype < self.config.init_temp_swap_prob + self.config.init_temp_ins_prob {
                        if let Some(a) = self.get_a_ins(curr_sol, aisles_vec, rng) { eval.try_apply_aisle_add(a, data); true } else { false }
                    } else {
                        if let Some(r) = self.get_a_rem(curr_sol, rng) { eval.try_apply_aisle_remove(r, data); true } else { false }
                    }
                }
            };

            if applied {
                let nobj = eval.current_objective();
                let delta = nobj - curr_obj;
                if delta < 0.0 {
                    deltas.push(delta.abs());
                }
                eval.rollback();
            }
        }

        if deltas.is_empty() {
            return self.config.t0; 
        }
        let avg_delta = deltas.iter().sum::<f64>() / deltas.len() as f64;
        -avg_delta / self.config.initial_acceptance_prob.ln()
    }

    fn get_block_swap(&self, sol: &ChallengeSolution, all: &[usize], data: &ProblemData, rng: &mut impl Rng) -> Option<(Vec<usize>, Vec<usize>)> {
        let active: Vec<usize> = sol.orders.ones().collect();
        if active.len() < 2 { return None; }
        
        let to_remove: Vec<usize> = active.choose_multiple(rng, 2).cloned().collect();
        
        let mut to_add = Vec::new();
        for _ in 0..self.config.max_attempts_block {
            let candidate = *all.choose(rng).unwrap();
            if !sol.orders.contains(candidate) && !to_add.contains(&candidate) {
                to_add.push(candidate);
                if to_add.len() == 2 { break; }
            }
        }
        
        if to_add.len() < 2 { return None; }
        
        let mut current_items: i64 = active.iter().map(|&i| data.order_total_items[i] as i64).sum();
        for &r in &to_remove { current_items -= data.order_total_items[r] as i64; }
        for &a in &to_add { current_items += data.order_total_items[a] as i64; }
        
        if current_items >= data.wave_size_lb as i64 && current_items <= data.wave_size_ub as i64 {
            Some((to_remove, to_add))
        } else {
            None
        }
    }

    fn get_swap(&self, sol: &ChallengeSolution, all: &[usize], rng: &mut impl Rng) -> Option<(usize, usize)> {
        let ins: Vec<usize> = sol.orders.ones().collect(); if ins.is_empty() { return None; }
        let r = *ins.choose(rng).unwrap();
        for _ in 0..self.config.max_attempts_swap { let c = *all.choose(rng).unwrap(); if !sol.orders.contains(c) { return Some((r, c)); } }
        None
    }
    fn get_ins(&self, sol: &ChallengeSolution, all: &[usize], data: &ProblemData, rng: &mut impl Rng) -> Option<usize> {
        for _ in 0..self.config.max_attempts_ins { let c = *all.choose(rng).unwrap(); if !sol.orders.contains(c) {
            let cur_it: u32 = sol.orders.ones().map(|i| data.order_total_items[i]).sum();
            if cur_it + data.order_total_items[c] <= data.wave_size_ub { return Some(c); }
        } }
        None
    }
    fn get_rem(&self, sol: &ChallengeSolution, data: &ProblemData, rng: &mut impl Rng) -> Option<usize> {
        let ins: Vec<usize> = sol.orders.ones().collect(); if ins.is_empty() { return None; }
        for _ in 0..self.config.max_attempts_rem { let c = *ins.choose(rng).unwrap(); 
            let cur_it: u32 = sol.orders.ones().map(|i| data.order_total_items[i]).sum();
            if cur_it - data.order_total_items[c] >= data.wave_size_lb { return Some(c); }
        }
        None
    }
    fn get_a_swap(&self, sol: &ChallengeSolution, all: &[usize], rng: &mut impl Rng) -> Option<(usize, usize)> {
        let ins: Vec<usize> = sol.aisles.ones().collect(); if ins.is_empty() { return None; }
        let r = *ins.choose(rng).unwrap();
        for _ in 0..self.config.max_attempts_swap { let c = *all.choose(rng).unwrap(); if !sol.aisles.contains(c) { return Some((r, c)); } }
        None
    }
    fn get_a_ins(&self, sol: &ChallengeSolution, all: &[usize], rng: &mut impl Rng) -> Option<usize> {
        for _ in 0..self.config.max_attempts_ins { let c = *all.choose(rng).unwrap(); if !sol.aisles.contains(c) { return Some(c); } }
        None
    }
    fn get_a_rem(&self, sol: &ChallengeSolution, rng: &mut impl Rng) -> Option<usize> {
        let ins: Vec<usize> = sol.aisles.ones().collect(); if ins.is_empty() { return None; }
        Some(*ins.choose(rng).unwrap())
    }
}
