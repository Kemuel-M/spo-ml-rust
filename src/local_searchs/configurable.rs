use crate::solution::{ChallengeSolution, ProblemData};
use crate::heuristics::LocalSearchAlgorithm;
use crate::evaluator::{Evaluator, StockBalanceEvaluator, Move};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use super::{SearchStrategy, NeighborhoodType, LocalSearchConfig, SearchDimension};

pub struct ConfigurableLocalSearch {
    pub config: LocalSearchConfig,
}

pub fn generate_neighborhood(config: &LocalSearchConfig, solution: &ChallengeSolution, data: &ProblemData) -> Vec<Move> {
    let mut rng = rand::rng();
    let mut moves = Vec::new();
    
    match config.dimension {
        SearchDimension::Orders => {
            let n_orders = data.orders.len();
            let orders_in: Vec<usize> = solution.orders.ones().collect();
            
            if orders_in.is_empty() && (config.neighborhood == NeighborhoodType::Swap || config.neighborhood == NeighborhoodType::Removal) {
                return moves;
            }

            match config.neighborhood {
                NeighborhoodType::Swap => {
                    let orders_out: Vec<usize> = (0..n_orders).filter(|idx| !solution.orders.contains(*idx)).collect();
                    if orders_out.is_empty() { return moves; }
                    
                    let total_possible = orders_in.len() * orders_out.len();
                    if total_possible > config.sampling_size {
                        for _ in 0..config.sampling_size {
                            let rem = *orders_in.choose(&mut rng).unwrap();
                            let add = *orders_out.choose(&mut rng).unwrap();
                            moves.push(Move::OrderSwap(rem, add));
                        }
                    } else {
                        for &rem in &orders_in {
                            for &add in &orders_out {
                                moves.push(Move::OrderSwap(rem, add));
                            }
                        }
                    }
                },
                NeighborhoodType::Insertion => {
                    let orders_out: Vec<usize> = (0..n_orders).filter(|idx| !solution.orders.contains(*idx)).collect();
                    if orders_out.is_empty() { return moves; }

                    if orders_out.len() > config.sampling_size {
                        let chosen = orders_out.choose_multiple(&mut rng, config.sampling_size);
                        for &add in chosen { moves.push(Move::OrderInsertion(add)); }
                    } else {
                        for &add in &orders_out { moves.push(Move::OrderInsertion(add)); }
                    }
                },
                NeighborhoodType::Removal => {
                    if orders_in.len() > config.sampling_size {
                        let chosen = orders_in.choose_multiple(&mut rng, config.sampling_size);
                        for &rem in chosen { moves.push(Move::OrderRemoval(rem)); }
                    } else {
                        for &rem in &orders_in { moves.push(Move::OrderRemoval(rem)); }
                    }
                }
            }
        },
        SearchDimension::Aisles => {
            let n_aisles = data.aisles.len();
            let aisles_in: Vec<usize> = solution.aisles.ones().collect();

            match config.neighborhood {
                NeighborhoodType::Swap => {
                    let aisles_out: Vec<usize> = (0..n_aisles).filter(|idx| !solution.aisles.contains(*idx)).collect();
                    if aisles_in.is_empty() || aisles_out.is_empty() { return moves; }

                    let total_possible = aisles_in.len() * aisles_out.len();
                    if total_possible > config.sampling_size {
                        for _ in 0..config.sampling_size {
                            let rem = *aisles_in.choose(&mut rng).unwrap();
                            let add = *aisles_out.choose(&mut rng).unwrap();
                            moves.push(Move::AisleSwap(rem, add));
                        }
                    } else {
                        for &rem in &aisles_in {
                            for &add in &aisles_out {
                                moves.push(Move::AisleSwap(rem, add));
                            }
                        }
                    }
                },
                NeighborhoodType::Insertion => {
                    let aisles_out: Vec<usize> = (0..n_aisles).filter(|idx| !solution.aisles.contains(*idx)).collect();
                    if aisles_out.is_empty() { return moves; }
                    if aisles_out.len() > config.sampling_size {
                        for &add in aisles_out.choose_multiple(&mut rng, config.sampling_size) {
                            moves.push(Move::AisleInsertion(add));
                        }
                    } else {
                        for &add in &aisles_out { moves.push(Move::AisleInsertion(add)); }
                    }
                },
                NeighborhoodType::Removal => {
                    if aisles_in.is_empty() { return moves; }
                    if aisles_in.len() > config.sampling_size {
                        for &rem in aisles_in.choose_multiple(&mut rng, config.sampling_size) {
                            moves.push(Move::AisleRemoval(rem));
                        }
                    } else {
                        for &rem in &aisles_in { moves.push(Move::AisleRemoval(rem)); }
                    }
                }
            }
        }
    }
    moves
}

impl LocalSearchAlgorithm for ConfigurableLocalSearch {
    fn name(&self) -> String {
        format!("{:?} using {:?} on {:?}", self.config.strategy, self.config.neighborhood, self.config.dimension)
    }

    fn refine(&self, solution: &mut ChallengeSolution, data: &ProblemData, seed: u64) -> bool {
        let mut evaluator: Box<dyn Evaluator> = Box::new(StockBalanceEvaluator::new(solution, data));
        
        // Sincroniza a solução inicial com o estado corrigido do evaluator
        solution.orders = evaluator.get_active_orders();
        solution.aisles = evaluator.get_active_aisles();
        solution.score = evaluator.current_objective();

        let initial_obj = evaluator.current_objective();
        let backup_solution = solution.clone();

        let mut global_improvement = false;
        let mut improved_iter = true;
        let mut iter_count = 0;
        let start_time = std::time::Instant::now();
        
        while improved_iter && iter_count < self.config.max_iterations {
            if start_time.elapsed().as_secs() >= self.config.max_time_secs {
                break;
            }
            iter_count += 1;
            improved_iter = false;
            
            match self.config.strategy {
                SearchStrategy::FirstImprovement => {
                    if self.run_first_improvement(&mut *evaluator, solution, data, seed + iter_count as u64) {
                        improved_iter = true;
                        global_improvement = true;
                    }
                },
                SearchStrategy::BestImprovement => {
                    if self.run_best_improvement(&mut *evaluator, solution, data) {
                        improved_iter = true;
                        global_improvement = true;
                    }
                },
            }
        }

        if evaluator.current_objective() < initial_obj - 1e-6 {
            *solution = backup_solution;
            return false;
        }
        
        // Garante que o score final esteja refletido
        solution.score = evaluator.current_objective();

        global_improvement
    }
}

impl ConfigurableLocalSearch {
    fn run_first_improvement(
        &self, 
        evaluator: &mut dyn Evaluator, 
        solution: &mut ChallengeSolution, 
        data: &ProblemData,
        seed: u64
    ) -> bool {
        let current_obj = evaluator.current_objective();
        let mut moves = generate_neighborhood(&self.config, solution, data);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        moves.shuffle(&mut rng);

        for mv in moves {
            if evaluator.validate_move(&mv, data) {
                if self.apply_move(evaluator, solution, data, mv, current_obj) {
                    return true;
                }
            }
        }
        false
    }

    fn run_best_improvement(
        &self,
        evaluator: &mut dyn Evaluator,
        solution: &mut ChallengeSolution,
        data: &ProblemData
    ) -> bool {
        let current_obj = evaluator.current_objective();
        let moves = generate_neighborhood(&self.config, solution, data);

        use rayon::prelude::*;
        
        let best_move_found = moves.into_par_iter()
            .map_init(
                || evaluator.clone_box(), 
                |local_eval, mv| {
                        if local_eval.validate_move(&mv, data) {
                            match mv {
                                Move::OrderSwap(o, i) => local_eval.try_apply_order_swap(o, i, data),
                                Move::OrderInsertion(i) => local_eval.try_apply_order_add(i, data),
                                Move::OrderRemoval(o) => local_eval.try_apply_order_remove(o, data),
                                Move::AisleSwap(o, i) => local_eval.try_apply_aisle_swap(o, i, data),
                                Move::AisleInsertion(i) => local_eval.try_apply_aisle_add(i, data),
                                Move::AisleRemoval(o) => local_eval.try_apply_aisle_remove(o, data),
                            };
                            let new_obj = local_eval.current_objective();
                            local_eval.rollback();
                            return Some((mv, new_obj));
                        }
                    None
                }
            )
            .filter_map(|x| x)
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        if let Some((mv, best_obj)) = best_move_found {
            if best_obj > current_obj + 1e-6 {
                return self.apply_move(evaluator, solution, data, mv, current_obj);
            }
        }
        false
    }

    fn apply_move(
        &self, 
        evaluator: &mut dyn Evaluator, 
        solution: &mut ChallengeSolution, 
        data: &ProblemData, 
        mv: Move, 
        current_obj: f64
    ) -> bool {
        match mv {
            Move::OrderSwap(o, i) => evaluator.try_apply_order_swap(o, i, data),
            Move::OrderInsertion(i) => evaluator.try_apply_order_add(i, data),
            Move::OrderRemoval(o) => evaluator.try_apply_order_remove(o, data),
            Move::AisleSwap(o, i) => evaluator.try_apply_aisle_swap(o, i, data),
            Move::AisleInsertion(i) => evaluator.try_apply_aisle_add(i, data),
            Move::AisleRemoval(o) => evaluator.try_apply_aisle_remove(o, data),
        };

        let new_obj = evaluator.current_objective();
        if new_obj > current_obj + 1e-6 {
            evaluator.commit();
            solution.orders = evaluator.get_active_orders();
            solution.aisles = evaluator.get_active_aisles();
            solution.score = new_obj;
            return true;
        }
        evaluator.rollback();
        false
    }
}
