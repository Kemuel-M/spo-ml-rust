import re
with open("src/evaluator/core_stock.rs", "r") as f:
    text = f.read()

text = text.replace(
    """    /// Lookahead: Verifica se a remoção do corredor deixa algum item negativo
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
    }""",
    """    /// Lookahead: Verifica se a remoção do corredor deixa algum item negativo
    #[inline(always)]
    pub fn can_remove_aisle_safely(&self, a: usize, data: &ProblemData) -> bool {
        for item in &data.dense_aisles[a] {
            if self.balance[item.id] - (item.qty as i32) < 0 {
                return false;
            }
        }
        true
    }"""
)

with open("src/evaluator/core_stock.rs", "w") as f:
    f.write(text)
