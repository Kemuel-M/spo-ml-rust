use crate::solution::ProblemData;

#[derive(Clone, Copy, Debug)]
pub enum Move {
    OrderSwap(usize, usize),
    OrderInsertion(usize),
    OrderRemoval(usize),
    AisleSwap(usize, usize),
    AisleInsertion(usize),
    AisleRemoval(usize),
}

pub trait Evaluator: Send + Sync {
    fn current_objective(&self) -> f64;
    fn is_stock_valid(&self) -> bool;
    fn current_total_items(&self) -> u32;

    fn try_apply_order_swap(&mut self, out_id: usize, in_id: usize, data: &ProblemData);
    fn try_apply_order_add(&mut self, in_id: usize, data: &ProblemData);
    fn try_apply_order_remove(&mut self, out_id: usize, data: &ProblemData);
    fn try_apply_aisle_swap(&mut self, out_id: usize, in_id: usize, data: &ProblemData);
    fn try_apply_aisle_add(&mut self, in_id: usize, data: &ProblemData);
    fn try_apply_aisle_remove(&mut self, out_id: usize, data: &ProblemData);

    fn rollback(&mut self);
    fn commit(&mut self);
    fn get_active_aisles(&self) -> fixedbitset::FixedBitSet;
    fn get_active_orders(&self) -> fixedbitset::FixedBitSet;
    
    fn sync_aisles_from_orders(&mut self, data: &ProblemData);
    fn sync_orders_from_aisles(&mut self, data: &ProblemData);
    
    fn sync_dimension(&mut self, dim: crate::local_searchs::SearchDimension, data: &ProblemData) {
        match dim {
            crate::local_searchs::SearchDimension::Orders => self.sync_aisles_from_orders(data),
            crate::local_searchs::SearchDimension::Aisles => self.sync_orders_from_aisles(data),
        }
    }

    fn clone_box(&self) -> Box<dyn Evaluator>;
    
    fn set_pruning(&mut self, _prune: bool) {} // Default trait impl
    
    // Antigo
    fn validate_move(&mut self, mv: &Move, data: &ProblemData) -> bool;
    
    // Novos otimizados
    fn test_move(&mut self, mv: &Move, data: &ProblemData) -> Option<f64>;
    fn test_and_apply_move(&mut self, mv: &Move, data: &ProblemData, min_obj: f64) -> bool;
}
