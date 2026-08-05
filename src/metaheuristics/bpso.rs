use crate::solution::{ChallengeSolution, ProblemData};
use crate::heuristics::{SolverStrategy, ConstructiveAlgorithm, LocalSearchAlgorithm, utils};
use crate::local_searchs::SearchDimension;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::time::{Instant, Duration};

use crate::solver::config::BpsoDimension;

pub struct BPSOConfig {
    pub dimension: BpsoDimension, pub population_size: usize, pub iterations: usize,
    pub w_min: f64, pub w_max: f64, pub c1: f64, pub c2: f64, 
    pub c1_escape: f64, pub c2_escape: f64,
    pub v_max: f64, pub ls_prob: f64, pub ls_prob_high: f64,
    pub max_time_secs: u64, pub stagnation_limit: usize, 
    pub stag_threshold_escape: f64, pub stag_threshold_panic: f64,
    pub p_orders: f64,
    pub turbulence_base: f64, pub turbulence_high: f64,
    pub score_threshold_orders: f64, pub score_threshold_aisles: f64,
    pub log_frequency: usize,
}

pub struct BPSO {
    pub constructive: Box<dyn ConstructiveAlgorithm>, pub local_search: Box<dyn LocalSearchAlgorithm>, pub config: BPSOConfig,
}

struct Particle {
    dimension: SearchDimension,
    position: Vec<bool>, velocity: Vec<f64>, best_position: Vec<bool>,
    best_fitness: f64, best_solution: ChallengeSolution, current_solution: ChallengeSolution,
    rng: ChaCha8Rng,
}

impl BPSO {
    fn sigmoid(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }
    fn solution_to_vec(sol: &ChallengeSolution, dim: SearchDimension, n: usize) -> Vec<bool> {
        let mut v = vec![false; n]; let bits = sol.bits(dim);
        for idx in bits.ones() { if idx < n { v[idx] = true; } } v
    }

    fn guided_construct(&self, data: &ProblemData, scores: &[f64], dim: SearchDimension, _seed: u64) -> ChallengeSolution {
        let n_orders = data.orders.len();
        if dim == SearchDimension::Orders {
            let mut sol = ChallengeSolution::new(n_orders, data.aisles.len());
            let mut surplus_stock = vec![0u32; data.n_items];
            let mut curr_items = 0;
            let mut current_obj = 0.0;

            let mut current_new_ac = data.order_initial_aisles_count.clone();

            let mut order_indices = data.all_order_indices.clone();
            order_indices.sort_by(|&a, &b| {
                scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal)
            });

            let mut unopened_req = Vec::with_capacity(32);
            for &idx in &order_indices {
                if curr_items >= data.wave_size_ub { break; }
                let total = data.order_total_items[idx];
                if curr_items + total > data.wave_size_ub { continue; }

                unopened_req.clear();
                for &a in &data.order_required_aisles[idx] {
                    if !sol.aisles.contains(a) { unopened_req.push(a); }
                }
                
                let mut can_fulfill = true;
                for item in &data.dense_orders[idx] {
                    let mut stock_we_will_have = surplus_stock[item.id];
                    for &a in &unopened_req {
                        stock_we_will_have += data.stock_matrix[a * data.n_items + item.id];
                    }
                    if stock_we_will_have < item.qty {
                        can_fulfill = false;
                        break;
                    }
                }
                if !can_fulfill { continue; }

                if curr_items >= data.wave_size_lb {
                    let new_ac_count = current_new_ac[idx];
                    let total_ac = sol.aisles.count_ones(..) + new_ac_count;
                    let new_obj = (curr_items + total) as f64 / total_ac as f64;
                    if new_obj <= current_obj && scores[idx] < self.config.score_threshold_orders { continue; }
                }

                sol.orders.insert(idx);
                curr_items += total;
                let mut newly_added_aisles = Vec::new();
                for &a in &data.order_required_aisles[idx] {
                    if !sol.aisles.contains(a) { 
                        sol.aisles.insert(a); 
                        newly_added_aisles.push(a); 
                        for item in &data.dense_aisles[a] {
                            surplus_stock[item.id] += item.qty;
                        }
                    }
                }

                for item in &data.dense_orders[idx] {
                    surplus_stock[item.id] -= item.qty;
                }
                for &a in &newly_added_aisles {
                    for &o_idx in &data.aisle_to_orders_req[a] { current_new_ac[o_idx] = current_new_ac[o_idx].saturating_sub(1); }
                }
                current_obj = curr_items as f64 / sol.aisles.count_ones(..) as f64;
            }
            sol.score = current_obj;
            sol.feasible = curr_items >= data.wave_size_lb;
            sol
        } else {
            // Aisle-Centric Guided Construct
            let mut sol = ChallengeSolution::new(n_orders, data.aisles.len());
            let mut total_curr_stock = vec![0u32; data.n_items];
            let mut total_items = 0;
            
            let mut aisle_indices: Vec<usize> = (0..data.aisles.len()).collect();
            aisle_indices.sort_by(|&a, &b| {
                scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal)
            });

            for &aid in &aisle_indices {
                if total_items >= data.wave_size_ub { break; }
                if scores[aid] < self.config.score_threshold_aisles { continue; }

                sol.aisles.insert(aid);
                for item in &data.dense_aisles[aid] {
                    total_curr_stock[item.id] += item.qty;
                }

                for &oid in &data.orders_sorted_by_size {
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
            sol.score = total_items as f64 / sol.aisles.count_ones(..).max(1) as f64;
            sol.feasible = total_items >= data.wave_size_lb && total_items <= data.wave_size_ub;
            sol
        }
    }
}

impl SolverStrategy for BPSO {
    fn name(&self) -> String { format!("BPSO ({:?}) [{} + {}]", self.config.dimension, self.constructive.name(), self.local_search.name()) }
    fn solve(&self, data: &ProblemData, seed: u64) -> ChallengeSolution {
        let start = Instant::now(); let max_dur = Duration::from_secs(self.config.max_time_secs);
        let n_orders = data.orders.len();
        let n_aisles = data.aisles.len();
        
        let mut particles: Vec<Particle> = (0..self.config.population_size).into_par_iter().map(|i| {
            let mut rng = ChaCha8Rng::seed_from_u64(seed + i as u64 + 1000);
            let dim = match self.config.dimension {
                BpsoDimension::Orders => SearchDimension::Orders,
                BpsoDimension::Aisles => SearchDimension::Aisles,
                BpsoDimension::Hybrid => if rng.random::<f64>() < self.config.p_orders { SearchDimension::Orders } else { SearchDimension::Aisles }
            };
            
            let s = self.constructive.construct(data, seed + i as u64); 
            let f = utils::compute_objective(&s, data);
            let n = if dim == SearchDimension::Orders { n_orders } else { n_aisles };
            let p = Self::solution_to_vec(&s, dim, n); 
            let mut v = vec![0.0; n];
            for val in v.iter_mut() { *val = rng.random_range(-self.config.v_max..self.config.v_max); }
            Particle { dimension: dim, position: p.clone(), velocity: v, best_position: p, best_fitness: f, best_solution: s.clone(), current_solution: s, rng }
        }).collect();

        let mut gbest_f = -1.0; 
        for p in &particles { if p.best_fitness > gbest_f { gbest_f = p.best_fitness; } }
        let gbest_idx = particles.iter().position(|p| p.best_fitness == gbest_f).unwrap_or(0);
        let mut gbest_s = particles[gbest_idx].best_solution.clone();
        let mut stagnation_count = 0;

        let mut gbest_v_orders = Self::solution_to_vec(&gbest_s, SearchDimension::Orders, n_orders);
        let mut gbest_v_aisles = Self::solution_to_vec(&gbest_s, SearchDimension::Aisles, n_aisles);

        let n_orders_particles = particles.iter().filter(|p| p.dimension == SearchDimension::Orders).count();
        let n_aisles_particles = particles.len() - n_orders_particles;

        println!("  {:<6} | {:<10} | {:<8} | {:<5} | {:<4} | {:<4} | {:<4} | {:<5} | {:<5} | {:<5} | {:<5} | {:<4}", 
                 "Iter", "Best Obj", "Avg Obj", "Imp%", "Stag", "w", "Turb", "AvgV", "LSHit", "VSat%", "HDst%", "Time");
        if self.config.dimension == BpsoDimension::Hybrid {
            println!("  [Hybrid Mode: {} Orders / {} Aisles]", n_orders_particles, n_aisles_particles);
        }
        println!("  {}", "-".repeat(95));

        let mut final_iterations = 0;
        let mut final_v_sat = 0.0;
        let mut final_h_dist = 0.0;

        for iter in 0..self.config.iterations {
            final_iterations = iter + 1;
            if start.elapsed() >= max_dur { break; }
            if stagnation_count >= self.config.stagnation_limit { break; }

            let w = self.config.w_max - (self.config.w_max - self.config.w_min) * (iter as f64 / self.config.iterations as f64);
            
            // --- INÍCIO: AR-BPSO (Adaptive Reactive) ---
            // Calcula o quão estagnado o algoritmo está (de 0.0 a 1.0)
            let stag_ratio = stagnation_count as f64 / self.config.stagnation_limit as f64;
            
            // Estágio 1 de Defesa: Alteração Psicológica (A partir de 30% de estagnação)
            let (current_c1, current_c2) = if stag_ratio > self.config.stag_threshold_escape {
                (self.config.c1_escape, self.config.c2_escape) // Modo Fuga (Enxame Teimoso e Explorador)
            } else {
                (self.config.c1, self.config.c2) // Modo Harmonia (1.494 - Convergência Rápida)
            };

            // Estágio 2 de Defesa: Caos Dinâmico e Busca Local Reativa (A partir de 60% de estagnação)
            let (turbulence_chance, current_ls_prob) = if stag_ratio > self.config.stag_threshold_panic {
                (self.config.turbulence_high, self.config.ls_prob_high) // Pânico
            } else {
                (self.config.turbulence_base, self.config.ls_prob) // Voo normal
            };
            // --- FIM: AR-BPSO ---

            // Convert gbest_s to both dimensions for the swarm to follow
            // gbest_v_orders and gbest_v_aisles are updated at the end of the loop if improved

            // Parallel computation of movement and stats
            let stats: (usize, f64, usize, usize, usize, usize, usize) = particles.par_iter_mut().map(|p| {
                let n = p.velocity.len();
                let gbest_v = if p.dimension == SearchDimension::Orders { &gbest_v_orders } else { &gbest_v_aisles };
                let mut scores = vec![0.0; n];
                let mut v_sum = 0.0;
                let mut v_sat_count = 0;
                let mut h_dist_count = 0;

                for j in 0..n {
                    let r1 = p.rng.random::<f64>(); let r2 = p.rng.random::<f64>();
                    let cog = current_c1 * r1 * (if p.best_position[j] { 1.0 } else { 0.0 } - if p.position[j] { 1.0 } else { 0.0 });
                    let soc = current_c2 * r2 * (if gbest_v[j] { 1.0 } else { 0.0 } - if p.position[j] { 1.0 } else { 0.0 });
                    p.velocity[j] = w * p.velocity[j] + cog + soc;
                    p.velocity[j] = p.velocity[j].clamp(-self.config.v_max, self.config.v_max);
                    v_sum += p.velocity[j].abs();
                    if p.velocity[j].abs() >= self.config.v_max * 0.95 { v_sat_count += 1; }
                    if p.position[j] != gbest_v[j] { h_dist_count += 1; }
                    scores[j] = Self::sigmoid(p.velocity[j]);
                }
                
                if p.rng.random::<f64>() < turbulence_chance {
                    for j in 0..n {
                        scores[j] = p.rng.random::<f64>(); 
                        p.velocity[j] = p.rng.random_range(-self.config.v_max..self.config.v_max);
                    }
                }
                
                let seed_construct = p.rng.next_u64();
                let mut sol = self.guided_construct(data, &scores, p.dimension, seed_construct);
                
                let mut ls_hit = 0;
                let mut ls_attempt = 0;
                if p.rng.random::<f64>() < current_ls_prob { 
                    ls_attempt = 1;
                    let before_f = utils::compute_objective(&sol, data);
                    let seed_ls = p.rng.next_u64();
                    self.local_search.refine(&mut sol, data, seed_ls); 
                    if utils::compute_objective(&sol, data) > before_f { ls_hit = 1; }
                }
                
                let f = utils::compute_objective(&sol, data);
                p.current_solution = sol;
                p.position = Self::solution_to_vec(&p.current_solution, p.dimension, n);
                
                let mut improved = 0;
                if f > p.best_fitness {
                    p.best_fitness = f; p.best_position = p.position.clone(); p.best_solution = p.current_solution.clone();
                    improved = 1;
                }
                (improved, v_sum / n as f64, ls_attempt, ls_hit, v_sat_count, h_dist_count, n)
            }).reduce(|| (0, 0.0, 0, 0, 0, 0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3, a.4 + b.4, a.5 + b.5, a.6 + b.6));

            let improvements = stats.0;
            let avg_v = stats.1 / particles.len() as f64;
            let ls_attempts = stats.2;
            let ls_hits = stats.3;
            let total_sat = stats.4;
            let total_h_dist = stats.5;
            let total_bits = stats.6;

            let avg_sat_percent = if total_bits > 0 { (total_sat as f64 / total_bits as f64) * 100.0 } else { 0.0 };
            let avg_h_dist_percent = if total_bits > 0 { (total_h_dist as f64 / total_bits as f64) * 100.0 } else { 0.0 };
            
            final_v_sat = avg_sat_percent;
            final_h_dist = avg_h_dist_percent;

            let avg_fitness: f64 = particles.iter().map(|p| p.best_fitness).sum::<f64>() / particles.len() as f64;
            let imp_percent = (improvements as f64 / particles.len() as f64) * 100.0;
            let ls_hit_rate = if ls_attempts > 0 { (ls_hits as f64 / ls_attempts as f64) * 100.0 } else { 0.0 };

            let mut improved = false;
            for p in &particles {
                if p.best_fitness > gbest_f {
                    gbest_f = p.best_fitness; gbest_s = p.best_solution.clone();
                    improved = true;
                }
            }

            if improved {
                gbest_v_orders = Self::solution_to_vec(&gbest_s, SearchDimension::Orders, n_orders);
                gbest_v_aisles = Self::solution_to_vec(&gbest_s, SearchDimension::Aisles, n_aisles);
                stagnation_count = 0; 
            } else { stagnation_count += 1; }
            if iter % self.config.log_frequency == 0 || improved {
                println!("  {:<6} | {:<10.4} | {:<8.4} | {:>4.1}% | {:<4} | {:<4.2} | {:<4.2} | {:<5.2} | {:>4.1}% | {:>4.1}% | {:>4.1}% | {:>3}s", 
                         iter, gbest_f, avg_fitness, imp_percent, stagnation_count, w, turbulence_chance, avg_v, ls_hit_rate, avg_sat_percent, avg_h_dist_percent, start.elapsed().as_secs());
            }
        }
        println!("  {}", "-".repeat(110));
        println!("  [BPSO] Finished in {} iterations. Best Obj: {:.4}", final_iterations, gbest_f);
        println!("  [BPSO Metrics] Final Velocity Saturation: {:.2}% | Final Diversity (HDist): {:.2}%", final_v_sat, final_h_dist);
        gbest_s
    }
}
