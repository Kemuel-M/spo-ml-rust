use crate::solution::{ChallengeSolution, ProblemData};
use crate::evaluator::Move;
use crate::local_searchs::{NeighborhoodType, SearchDimension};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

pub trait Neighborhood: Send + Sync {
    fn generate_moves(&self, solution: &ChallengeSolution, data: &ProblemData, sampling_size: usize) -> Vec<Move>;
    fn get_random_move(&self, solution: &ChallengeSolution, data: &ProblemData, rng: &mut ChaCha8Rng) -> Option<Move>;
    fn name(&self) -> String;
}

pub fn build_neighborhood(dimension: SearchDimension, neighborhood: NeighborhoodType) -> Box<dyn Neighborhood> {
    match dimension {
        SearchDimension::Orders => match neighborhood {
            NeighborhoodType::Swap => Box::new(OrderSwapNeighborhood),
            NeighborhoodType::Insertion => Box::new(OrderInsertionNeighborhood),
            NeighborhoodType::Removal => Box::new(OrderRemovalNeighborhood),
        },
        SearchDimension::Aisles => match neighborhood {
            NeighborhoodType::Swap => Box::new(AisleSwapNeighborhood),
            NeighborhoodType::Insertion => Box::new(AisleInsertionNeighborhood),
            NeighborhoodType::Removal => Box::new(AisleRemovalNeighborhood),
        },
    }
}

pub struct OrderSwapNeighborhood;
impl Neighborhood for OrderSwapNeighborhood {
    fn generate_moves(&self, solution: &ChallengeSolution, data: &ProblemData, sampling_size: usize) -> Vec<Move> {
        let n_orders = data.orders.len();
        let orders_in: Vec<usize> = solution.orders.ones().collect();
        let mut moves = Vec::new();
        if orders_in.is_empty() { return moves; }
        
        let orders_out: Vec<usize> = (0..n_orders).filter(|idx| !solution.orders.contains(*idx)).collect();
        if orders_out.is_empty() { return moves; }
        
        let total_possible = orders_in.len() * orders_out.len();
        if total_possible > sampling_size {
            let mut rng = rand::rng();
            for _ in 0..sampling_size {
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
        moves
    }
    
    fn get_random_move(&self, solution: &ChallengeSolution, data: &ProblemData, rng: &mut ChaCha8Rng) -> Option<Move> {
        let n_orders = data.orders.len();
        let orders_in: Vec<usize> = solution.orders.ones().collect();
        if orders_in.is_empty() { return None; }
        if solution.orders.count_ones(..) == n_orders { return None; }
        
        let rem = *orders_in.choose(rng).unwrap();
        let mut add = rng.random_range(0..n_orders);
        while solution.orders.contains(add) { add = rng.random_range(0..n_orders); }
        Some(Move::OrderSwap(rem, add))
    }
    
    fn name(&self) -> String { "Order Swap".to_string() }
}

pub struct OrderInsertionNeighborhood;
impl Neighborhood for OrderInsertionNeighborhood {
    fn generate_moves(&self, solution: &ChallengeSolution, data: &ProblemData, sampling_size: usize) -> Vec<Move> {
        let n_orders = data.orders.len();
        let mut moves = Vec::new();
        let orders_out: Vec<usize> = (0..n_orders).filter(|idx| !solution.orders.contains(*idx)).collect();
        if orders_out.is_empty() { return moves; }

        if orders_out.len() > sampling_size {
            let mut rng = rand::rng();
            let chosen = orders_out.choose_multiple(&mut rng, sampling_size);
            for &add in chosen { moves.push(Move::OrderInsertion(add)); }
        } else {
            for &add in &orders_out { moves.push(Move::OrderInsertion(add)); }
        }
        moves
    }
    
    fn get_random_move(&self, solution: &ChallengeSolution, data: &ProblemData, rng: &mut ChaCha8Rng) -> Option<Move> {
        let n_orders = data.orders.len();
        if solution.orders.count_ones(..) == n_orders { return None; }
        let mut add = rng.random_range(0..n_orders);
        while solution.orders.contains(add) { add = rng.random_range(0..n_orders); }
        Some(Move::OrderInsertion(add))
    }
    
    fn name(&self) -> String { "Order Insertion".to_string() }
}

pub struct OrderRemovalNeighborhood;
impl Neighborhood for OrderRemovalNeighborhood {
    fn generate_moves(&self, solution: &ChallengeSolution, _data: &ProblemData, sampling_size: usize) -> Vec<Move> {
        let mut moves = Vec::new();
        let orders_in: Vec<usize> = solution.orders.ones().collect();
        if orders_in.is_empty() { return moves; }

        if orders_in.len() > sampling_size {
            let mut rng = rand::rng();
            let chosen = orders_in.choose_multiple(&mut rng, sampling_size);
            for &rem in chosen { moves.push(Move::OrderRemoval(rem)); }
        } else {
            for &rem in &orders_in { moves.push(Move::OrderRemoval(rem)); }
        }
        moves
    }
    
    fn get_random_move(&self, solution: &ChallengeSolution, _data: &ProblemData, rng: &mut ChaCha8Rng) -> Option<Move> {
        let orders_in: Vec<usize> = solution.orders.ones().collect();
        if orders_in.is_empty() { return None; }
        let rem = *orders_in.choose(rng).unwrap();
        Some(Move::OrderRemoval(rem))
    }
    
    fn name(&self) -> String { "Order Removal".to_string() }
}

pub struct AisleSwapNeighborhood;
impl Neighborhood for AisleSwapNeighborhood {
    fn generate_moves(&self, solution: &ChallengeSolution, data: &ProblemData, sampling_size: usize) -> Vec<Move> {
        let n_aisles = data.aisles.len();
        let aisles_in: Vec<usize> = solution.aisles.ones().collect();
        let mut moves = Vec::new();
        
        let aisles_out: Vec<usize> = (0..n_aisles).filter(|idx| !solution.aisles.contains(*idx)).collect();
        if aisles_in.is_empty() || aisles_out.is_empty() { return moves; }

        let total_possible = aisles_in.len() * aisles_out.len();
        if total_possible > sampling_size {
            let mut rng = rand::rng();
            for _ in 0..sampling_size {
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
        moves
    }
    
    fn get_random_move(&self, solution: &ChallengeSolution, data: &ProblemData, rng: &mut ChaCha8Rng) -> Option<Move> {
        let n_aisles = data.aisles.len();
        let aisles_in: Vec<usize> = solution.aisles.ones().collect();
        if aisles_in.is_empty() { return None; }
        if solution.aisles.count_ones(..) == n_aisles { return None; }
        
        let rem = *aisles_in.choose(rng).unwrap();
        let mut add = rng.random_range(0..n_aisles);
        while solution.aisles.contains(add) { add = rng.random_range(0..n_aisles); }
        Some(Move::AisleSwap(rem, add))
    }
    
    fn name(&self) -> String { "Aisle Swap".to_string() }
}

pub struct AisleInsertionNeighborhood;
impl Neighborhood for AisleInsertionNeighborhood {
    fn generate_moves(&self, solution: &ChallengeSolution, data: &ProblemData, sampling_size: usize) -> Vec<Move> {
        let n_aisles = data.aisles.len();
        let mut moves = Vec::new();
        let aisles_out: Vec<usize> = (0..n_aisles).filter(|idx| !solution.aisles.contains(*idx)).collect();
        if aisles_out.is_empty() { return moves; }
        
        if aisles_out.len() > sampling_size {
            let mut rng = rand::rng();
            for &add in aisles_out.choose_multiple(&mut rng, sampling_size) {
                moves.push(Move::AisleInsertion(add));
            }
        } else {
            for &add in &aisles_out { moves.push(Move::AisleInsertion(add)); }
        }
        moves
    }
    
    fn get_random_move(&self, solution: &ChallengeSolution, data: &ProblemData, rng: &mut ChaCha8Rng) -> Option<Move> {
        let n_aisles = data.aisles.len();
        if solution.aisles.count_ones(..) == n_aisles { return None; }
        let mut add = rng.random_range(0..n_aisles);
        while solution.aisles.contains(add) { add = rng.random_range(0..n_aisles); }
        Some(Move::AisleInsertion(add))
    }
    
    fn name(&self) -> String { "Aisle Insertion".to_string() }
}

pub struct AisleRemovalNeighborhood;
impl Neighborhood for AisleRemovalNeighborhood {
    fn generate_moves(&self, solution: &ChallengeSolution, _data: &ProblemData, sampling_size: usize) -> Vec<Move> {
        let mut moves = Vec::new();
        let aisles_in: Vec<usize> = solution.aisles.ones().collect();
        if aisles_in.is_empty() { return moves; }
        
        if aisles_in.len() > sampling_size {
            let mut rng = rand::rng();
            for &rem in aisles_in.choose_multiple(&mut rng, sampling_size) {
                moves.push(Move::AisleRemoval(rem));
            }
        } else {
            for &rem in &aisles_in { moves.push(Move::AisleRemoval(rem)); }
        }
        moves
    }
    
    fn get_random_move(&self, solution: &ChallengeSolution, _data: &ProblemData, rng: &mut ChaCha8Rng) -> Option<Move> {
        let aisles_in: Vec<usize> = solution.aisles.ones().collect();
        if aisles_in.is_empty() { return None; }
        let rem = *aisles_in.choose(rng).unwrap();
        Some(Move::AisleRemoval(rem))
    }
    
    fn name(&self) -> String { "Aisle Removal".to_string() }
}
