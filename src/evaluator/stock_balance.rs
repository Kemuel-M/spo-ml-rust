use crate::solution::{ChallengeSolution, ProblemData};
use super::types::{Evaluator, Move};
use fixedbitset::FixedBitSet;
use super::core_stock::GlobalStock;

#[derive(Clone)]
struct Change {
    idx: usize,
    old_balance: i32,
}

#[derive(Clone)]
pub struct StockBalanceEvaluator {
    pub orders: FixedBitSet,
    pub aisles: FixedBitSet,
    pub total_items: u32,
    pub stock: GlobalStock,
    violations: usize,
    violating_items: FixedBitSet,
    pub wave_size_lb: u32,
    pub wave_size_ub: u32,
    pub prune_active: bool,
    
    // Estado para Rollback
    last_orders: FixedBitSet,
    last_aisles: FixedBitSet,
    last_total_items: u32,
    last_violations: usize,
    last_violating_items: FixedBitSet,
    
    // Log de alterações cirúrgico
    change_log: Vec<Change>,
}

impl StockBalanceEvaluator {
    pub fn new(solution: &ChallengeSolution, data: &ProblemData) -> Self {
        let n_items = data.n_items;
        let stock = GlobalStock::from_solution(solution, data);
        let mut total_items = 0;
        for o in solution.orders.ones() {
            total_items += data.order_total_items[o];
        }
        let mut vit = FixedBitSet::with_capacity(n_items);
        let mut v = 0;
        for (i, &b) in stock.balance.iter().enumerate() { if b < 0 { v += 1; vit.insert(i); } }

        Self {
            orders: solution.orders.clone(), aisles: solution.aisles.clone(), total_items,
            stock, violations: v, violating_items: vit,
            wave_size_lb: data.wave_size_lb, wave_size_ub: data.wave_size_ub,
            prune_active: true,
            last_orders: solution.orders.clone(), last_aisles: solution.aisles.clone(),
            last_total_items: total_items, last_violations: v,
            last_violating_items: FixedBitSet::with_capacity(n_items), 
            change_log: Vec::with_capacity(256), 
        }.init_state()
    }

    fn init_state(mut self) -> Self {
        self.last_violating_items = self.violating_items.clone();
        self.commit();
        self
    }

    fn update_order(&mut self, o: usize, add: bool, data: &ProblemData) {
        let items = &data.dense_orders[o];
        let mut sum = 0;
        for item in items {
            self.change_log.push(Change { idx: item.id, old_balance: self.stock.balance[item.id] });
            
            let b = &mut self.stock.balance[item.id];
            if add {
                if *b >= 0 && *b - (item.qty as i32) < 0 { self.violations += 1; self.violating_items.insert(item.id); }
                *b -= item.qty as i32;
            } else {
                if *b < 0 && *b + (item.qty as i32) >= 0 { self.violations -= 1; self.violating_items.remove(item.id); }
                *b += item.qty as i32;
            }
            sum += item.qty;
        }
        if add { self.total_items += sum; } else { self.total_items -= sum; }
    }

    fn update_aisle(&mut self, a: usize, add: bool, data: &ProblemData) {
        for item in &data.dense_aisles[a] {
            self.change_log.push(Change { idx: item.id, old_balance: self.stock.balance[item.id] });

            let b = &mut self.stock.balance[item.id];
            if add {
                if *b < 0 && *b + (item.qty as i32) >= 0 { self.violations -= 1; self.violating_items.remove(item.id); }
                *b += item.qty as i32;
            } else {
                if *b >= 0 && *b - (item.qty as i32) < 0 { self.violations += 1; self.violating_items.insert(item.id); }
                *b -= item.qty as i32;
            }
        }
    }
}

impl Evaluator for StockBalanceEvaluator {
    fn current_objective(&self) -> f64 {
        let ac = self.aisles.count_ones(..);
        if ac == 0 { return 0.0; }
        let raw = self.total_items as f64 / ac as f64;
        if self.violations > 0 { return 0.0; }
        let mut pen = 0.0;
        if self.total_items < self.wave_size_lb { pen = (self.wave_size_lb - self.total_items) as f64 * 5.0; }
        else if self.total_items > self.wave_size_ub { pen = (self.total_items - self.wave_size_ub) as f64 * 5.0; }
        (raw - pen).max(0.0)
    }
    fn is_stock_valid(&self) -> bool { self.violations == 0 }

    fn current_total_items(&self) -> u32 { self.total_items }
    fn get_active_aisles(&self) -> FixedBitSet { self.aisles.clone() }
    fn get_active_orders(&self) -> FixedBitSet { self.orders.clone() }

    fn try_apply_order_swap(&mut self, out: usize, in_id: usize, data: &ProblemData) {
        self.update_order(out, false, data); self.orders.remove(out);
        self.update_order(in_id, true, data); self.orders.insert(in_id);
        self.sync_aisles_from_orders(data);
    }
    fn try_apply_order_add(&mut self, in_id: usize, data: &ProblemData) {
        self.update_order(in_id, true, data); self.orders.insert(in_id);
        self.sync_aisles_from_orders(data);
    }
    fn try_apply_order_remove(&mut self, out: usize, data: &ProblemData) {
        self.update_order(out, false, data); self.orders.remove(out);
        self.sync_aisles_from_orders(data);
    }
    fn try_apply_aisle_swap(&mut self, out: usize, in_id: usize, data: &ProblemData) {
        self.update_aisle(out, false, data); self.aisles.remove(out);
        self.update_aisle(in_id, true, data); self.aisles.insert(in_id);
        self.sync_orders_from_aisles(data);
    }
    fn try_apply_aisle_add(&mut self, in_id: usize, data: &ProblemData) {
        self.update_aisle(in_id, true, data); self.aisles.insert(in_id);
        self.sync_orders_from_aisles(data);
    }
    fn try_apply_aisle_remove(&mut self, out: usize, data: &ProblemData) {
        self.update_aisle(out, false, data); self.aisles.remove(out);
        self.sync_orders_from_aisles(data);
    }

    fn rollback(&mut self) {
        while let Some(change) = self.change_log.pop() {
            self.stock.balance[change.idx] = change.old_balance;
        }
        self.orders.clone_from(&self.last_orders); 
        self.aisles.clone_from(&self.last_aisles);
        self.total_items = self.last_total_items; 
        self.violations = self.last_violations; 
        self.violating_items.clone_from(&self.last_violating_items);
    }
    
    fn commit(&mut self) {
        self.change_log.clear();
        self.last_orders.clone_from(&self.orders); 
        self.last_aisles.clone_from(&self.aisles);
        self.last_total_items = self.total_items; 
        self.last_violations = self.violations; 
        self.last_violating_items.clone_from(&self.violating_items);
    }

    fn sync_aisles_from_orders(&mut self, data: &ProblemData) {
        while self.violations > 0 {
            let mut best_aisle = None; let mut max_red = 0;
            if let Some(it_id) = self.violating_items.ones().next() {
                for a_idx in data.item_locations_bits[it_id].ones() {
                    if self.aisles.contains(a_idx) { continue; }
                    let mut red = 0;
                    for a_item in &data.dense_aisles[a_idx] {
                        let cb = self.stock.balance[a_item.id];
                        if cb < 0 { red += std::cmp::min(a_item.qty as i32, -cb); }
                    }
                    if red > max_red { max_red = red; best_aisle = Some(a_idx); }
                }
            }
            if let Some(a_idx) = best_aisle { self.update_aisle(a_idx, true, data); self.aisles.insert(a_idx); } else { break; }
        }
        if self.prune_active {
            let active: Vec<usize> = self.aisles.ones().collect();
            for a_idx in active {
                if self.stock.can_remove_aisle_safely(a_idx, data) {
                    self.update_aisle(a_idx, false, data); self.aisles.remove(a_idx);
                }
            }
        }
    }

    fn sync_orders_from_aisles(&mut self, data: &ProblemData) {
        while self.violations > 0 && self.orders.count_ones(..) > 0 {
            let vit = self.violating_items.ones().next().unwrap();
            let mut orem = None;
            for o_idx in self.orders.ones() {
                for item in &data.dense_orders[o_idx] { if item.id == vit { orem = Some(o_idx); break; } }
                if orem.is_some() { break; }
            }
            if let Some(o_idx) = orem { self.update_order(o_idx, false, data); self.orders.remove(o_idx); } else { break; }
        }
    }

    fn validate_move(&mut self, mv: &Move, data: &ProblemData) -> bool {
        match *mv {
            Move::OrderInsertion(i) => if self.total_items + data.order_total_items[i] > self.wave_size_ub { return false; },
            Move::OrderSwap(o, i) => if self.total_items - data.order_total_items[o] + data.order_total_items[i] > self.wave_size_ub { return false; },
            Move::OrderRemoval(o) => if self.total_items < data.order_total_items[o] + self.wave_size_lb { return false; },
            _ => {}
        }
        match *mv {
            Move::OrderSwap(o, i) => { if self.total_items - data.order_total_items[o] + data.order_total_items[i] < self.wave_size_lb { return false; } },
            _ => {}
        }
        
        match *mv {
            Move::OrderSwap(o, i) => self.try_apply_order_swap(o, i, data),
            Move::OrderInsertion(i) => self.try_apply_order_add(i, data),
            Move::OrderRemoval(o) => self.try_apply_order_remove(o, data),
            Move::AisleSwap(o, i) => self.try_apply_aisle_swap(o, i, data),
            Move::AisleInsertion(i) => self.try_apply_aisle_add(i, data),
            Move::AisleRemoval(o) => self.try_apply_aisle_remove(o, data),
        }
        let ok = self.is_stock_valid() && self.total_items >= self.wave_size_lb && self.total_items <= self.wave_size_ub;
        self.rollback(); 
        ok
    }
    
    fn test_move(&mut self, mv: &Move, data: &ProblemData) -> Option<f64> {
        match *mv {
            Move::OrderInsertion(i) => if self.total_items + data.order_total_items[i] > self.wave_size_ub { return None; },
            Move::OrderSwap(o, i) => if self.total_items - data.order_total_items[o] + data.order_total_items[i] > self.wave_size_ub { return None; },
            Move::OrderRemoval(o) => if self.total_items < data.order_total_items[o] + self.wave_size_lb { return None; },
            _ => {}
        }
        match *mv {
            Move::OrderSwap(o, i) => { if self.total_items - data.order_total_items[o] + data.order_total_items[i] < self.wave_size_lb { return None; } },
            _ => {}
        }

        match *mv {
            Move::OrderSwap(o, i) => self.try_apply_order_swap(o, i, data),
            Move::OrderInsertion(i) => self.try_apply_order_add(i, data),
            Move::OrderRemoval(o) => self.try_apply_order_remove(o, data),
            Move::AisleSwap(o, i) => self.try_apply_aisle_swap(o, i, data),
            Move::AisleInsertion(i) => self.try_apply_aisle_add(i, data),
            Move::AisleRemoval(o) => self.try_apply_aisle_remove(o, data),
        }
        
        let ok = self.is_stock_valid() && self.total_items >= self.wave_size_lb && self.total_items <= self.wave_size_ub;
        let mut score = None;
        if ok { score = Some(self.current_objective()); }
        self.rollback();
        score
    }
    
    fn test_and_apply_move(&mut self, mv: &Move, data: &ProblemData, min_obj: f64) -> bool {
        match *mv {
            Move::OrderInsertion(i) => if self.total_items + data.order_total_items[i] > self.wave_size_ub { return false; },
            Move::OrderSwap(o, i) => if self.total_items - data.order_total_items[o] + data.order_total_items[i] > self.wave_size_ub { return false; },
            Move::OrderRemoval(o) => if self.total_items < data.order_total_items[o] + self.wave_size_lb { return false; },
            _ => {}
        }
        match *mv {
            Move::OrderSwap(o, i) => { if self.total_items - data.order_total_items[o] + data.order_total_items[i] < self.wave_size_lb { return false; } },
            _ => {}
        }

        match *mv {
            Move::OrderSwap(o, i) => self.try_apply_order_swap(o, i, data),
            Move::OrderInsertion(i) => self.try_apply_order_add(i, data),
            Move::OrderRemoval(o) => self.try_apply_order_remove(o, data),
            Move::AisleSwap(o, i) => self.try_apply_aisle_swap(o, i, data),
            Move::AisleInsertion(i) => self.try_apply_aisle_add(i, data),
            Move::AisleRemoval(o) => self.try_apply_aisle_remove(o, data),
        }
        
        let ok = self.is_stock_valid() && self.total_items >= self.wave_size_lb && self.total_items <= self.wave_size_ub;
        if ok {
            let score = self.current_objective();
            if score > min_obj + 1e-6 {
                self.prune_active = true;
                self.sync_aisles_from_orders(data); 
                self.prune_active = false;
                
                let final_score = self.current_objective();
                if final_score > min_obj + 1e-6 {
                    self.commit();
                    return true;
                }
            }
        }
        self.rollback();
        false
    }
    
    fn set_pruning(&mut self, prune: bool) {
        self.prune_active = prune;
    }

    fn clone_box(&self) -> Box<dyn Evaluator> { Box::new(self.clone()) }
}
