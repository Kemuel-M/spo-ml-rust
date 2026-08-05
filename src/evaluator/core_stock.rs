use crate::solution::{ChallengeSolution, ProblemData};

#[derive(Clone)]
pub struct GlobalStock {
    pub balance: Vec<i32>,
    pub n_items: usize,
}

impl GlobalStock {
    pub fn new(n_items: usize) -> Self {
        Self {
            balance: vec![0; n_items],
            n_items,
        }
    }

    pub fn from_solution(solution: &ChallengeSolution, data: &ProblemData) -> Self {
        let mut gs = Self::new(data.n_items);
        for a in solution.aisles.ones() { gs.add_aisle_sparse(a, data); }
        for o in solution.orders.ones() { gs.remove_order_sparse(o, data); }
        gs
    }

    /// O(N_ITEMS) Auto-Vectorized by LLVM (AVX2/SIMD)
    #[inline(always)]
    pub fn add_aisle_dense(&mut self, a: usize, data: &ProblemData) {
        let start = a * self.n_items;
        let aisle_stock = &data.stock_matrix[start .. start + self.n_items];
        let balance = &mut self.balance;
        // The compiler will vectorize this loop natively without branches
        for i in 0..self.n_items {
            balance[i] += aisle_stock[i] as i32;
        }
    }

    #[inline(always)]
    pub fn add_aisle_sparse(&mut self, a: usize, data: &ProblemData) {
        for item in &data.dense_aisles[a] {
            self.balance[item.id] += item.qty as i32;
        }
    }

    #[inline(always)]
    pub fn remove_aisle_dense(&mut self, a: usize, data: &ProblemData) {
        let start = a * self.n_items;
        let aisle_stock = &data.stock_matrix[start .. start + self.n_items];
        let balance = &mut self.balance;
        for i in 0..self.n_items {
            balance[i] -= aisle_stock[i] as i32;
        }
    }

    #[inline(always)]
    pub fn remove_order_sparse(&mut self, o: usize, data: &ProblemData) {
        for item in &data.dense_orders[o] {
            if item.id < self.n_items {
                self.balance[item.id] -= item.qty as i32;
            }
        }
    }

    #[inline(always)]
    pub fn add_order_sparse(&mut self, o: usize, data: &ProblemData) {
        for item in &data.dense_orders[o] {
            if item.id < self.n_items {
                self.balance[item.id] += item.qty as i32;
            }
        }
    }

    /// Lookahead: Verifica se a remoção do corredor deixa algum item negativo
    #[inline(always)]
    pub fn can_remove_aisle_safely(&self, a: usize, data: &ProblemData) -> bool {
        let start = a * self.n_items;
        let aisle_stock = &data.stock_matrix[start .. start + self.n_items];
        for i in 0..self.n_items {
            if self.balance[i] - (aisle_stock[i] as i32) < 0 {
                return false;
            }
        }
        true
    }

    /// Lookahead: Verifica se a adição de um pedido excede o estoque atual
    #[inline(always)]
    pub fn can_add_order_safely(&self, o: usize, data: &ProblemData) -> bool {
        for item in &data.dense_orders[o] {
            if item.id < self.n_items && self.balance[item.id] - (item.qty as i32) < 0 {
                return false;
            }
        }
        true
    }

    /// Para as heurísticas construtivas: verifica se um pedido pode ser atendido
    /// considerando o estoque atual mais o estoque dos corredores requeridos que ainda não estão abertos.
    #[inline(always)]
    pub fn can_fulfill_order_with_aisles(&self, o: usize, unopened_req: &[usize], data: &ProblemData) -> bool {
        for item in &data.dense_orders[o] {
            if item.id < self.n_items {
                let mut stock_we_will_have = self.balance[item.id];
                for &a in unopened_req {
                    stock_we_will_have += data.stock_matrix[a * self.n_items + item.id] as i32;
                }
                if stock_we_will_have < item.qty as i32 {
                    return false;
                }
            }
        }
        true
    }
}
