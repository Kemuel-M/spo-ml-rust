use crate::solution::{ChallengeSolution, ProblemData};
use super::ConstructiveAlgorithm;
use super::utils;
use rand::prelude::*;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

pub struct RandomGreedy {
    pub alpha: f64,
}

use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Clone, Copy)]
struct Candidate {
    id: usize,
    score: f64,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool { self.id == other.id }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.score.partial_cmp(&other.score)
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

impl ConstructiveAlgorithm for RandomGreedy {
    fn name(&self) -> String { format!("Random Greedy (alpha={:.2})", self.alpha) }

    fn with_alpha(&self, alpha: f64) -> Option<Box<dyn ConstructiveAlgorithm>> {
        Some(Box::new(RandomGreedy { alpha }))
    }

    fn construct(&self, data: &ProblemData, seed: u64) -> ChallengeSolution {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let n_orders = data.orders.len();
        let mut solution = ChallengeSolution::new(n_orders, data.aisles.len());
        let mut available_stock = data.stock_matrix.clone();
        let mut curr_items = 0;
        let mut current_obj = 0.0;

        // Pré-calculo de novos corredores necessários (dinâmico)
        let mut current_new_ac: Vec<usize> = data.order_required_aisles.iter().map(|req| req.len()).collect();

        // Heap para manter os candidatos ordenados
        let mut heap = BinaryHeap::with_capacity(n_orders);
        for i in 0..n_orders {
            let total = data.order_total_items[i];
            let new_ac = current_new_ac[i];
            let score = if new_ac == 0 { f64::MAX / 2.0 + total as f64 } else { total as f64 / (new_ac as f64).powi(2) };
            heap.push(Candidate { id: i, score });
        }

        while curr_items < data.wave_size_ub && !heap.is_empty() {
            // Pegamos um conjunto de melhores candidatos do RCL (Restricted Candidate List) baseado no alpha
            let mut rcl = Vec::with_capacity(32);
            
            // Extrai candidatos potenciais do topo do heap
            while rcl.len() < 20 && !heap.is_empty() {
                let cand = heap.pop().unwrap();
                let total = data.order_total_items[cand.id];
                
                // Poda simples por limite UB
                if curr_items + total > data.wave_size_ub { continue; }
                
                // Lazy update do score: se o número de corredores mudou, o score no heap está desatualizado
                let actual_new_ac = current_new_ac[cand.id];
                let actual_score = if actual_new_ac == 0 { f64::MAX / 2.0 + total as f64 } else { total as f64 / (actual_new_ac as f64).powi(2) };
                
                if (cand.score - actual_score).abs() > 1e-6 {
                    // Score mudou drasticamente, re-insere e continua
                    heap.push(Candidate { id: cand.id, score: actual_score });
                    continue;
                }

                // Verifica estoque de forma lazy
                if utils::is_stock_sufficient_dense(&data.dense_orders[cand.id], &data.order_required_aisles[cand.id], &available_stock, data.n_items) {
                    rcl.push(cand);
                } else {
                    // Candidato inválido por estoque agora, mas guardamos para não perder (pode voltar se corredores mudarem? Não, stock só diminui)
                    // Não re-inserimos candidatos sem estoque.
                }
            }

            if rcl.is_empty() { break; }

            // Escolha aleatória dentro da fatia de qualidade (alpha)
            let chosen_idx = if self.alpha == 0.0 { 0 } else { rng.random_range(0..((rcl.len() as f64 * self.alpha).max(1.0) as usize).min(rcl.len())) };
            let chosen = rcl.remove(chosen_idx);
            
            // Devolve os outros candidatos não escolhidos para o heap
            for remaining in rcl { heap.push(remaining); }

            let idx = chosen.id;
            let total = data.order_total_items[idx];
            let req = &data.order_required_aisles[idx];

            // Critério de melhoria do objetivo (só aplica se já atingiu LB)
            if curr_items >= data.wave_size_lb {
                let new_ac_count = current_new_ac[idx];
                let total_ac = solution.aisles.count_ones(..) + new_ac_count;
                let new_obj = (curr_items + total) as f64 / total_ac as f64;
                if new_obj <= current_obj {
                    continue; // Pula este candidato mas não encerra a construção
                }
            }

            // Aplica a escolha
            solution.orders.insert(idx);
            curr_items += total;
            
            let mut newly_added_aisles = Vec::new();
            for &a in req {
                if !solution.aisles.contains(a) {
                    solution.aisles.insert(a);
                    newly_added_aisles.push(a);
                }
            }

            // Atualiza estoque disponível
            utils::update_stock_dense(&data.dense_orders[idx], &data.item_to_aisles, &mut available_stock, &solution.aisles, data.n_items);
            
            // Atualiza o custo incremental (corredores) de todas as outras ordens afetadas
            for &a in &newly_added_aisles {
                for &o_idx in &data.aisle_to_orders_req[a] {
                    current_new_ac[o_idx] = current_new_ac[o_idx].saturating_sub(1);
                    // O score no heap ficará desatualizado, o que tratamos com o lazy update acima
                }
            }
            
            current_obj = curr_items as f64 / solution.aisles.count_ones(..) as f64;
        }
        
        solution.score = current_obj;
        solution.feasible = curr_items >= data.wave_size_lb && curr_items <= data.wave_size_ub;
        solution
    }
}
