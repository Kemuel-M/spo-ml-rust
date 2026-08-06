import re
with open("src/evaluator/stock_balance.rs", "r") as f:
    text = f.read()

text = text.replace(
    """    fn validate_move(&mut self, mv: &Move, data: &ProblemData) -> bool {
        // Lookahead para movimentos destrutivos que não dependem de reparo""",
    """    fn validate_move(&mut self, mv: &Move, data: &ProblemData) -> bool {
        let t_start = std::time::Instant::now();
        // Lookahead para movimentos destrutivos que não dependem de reparo"""
)

text = text.replace(
    """        let ok = self.is_stock_valid() && self.total_items >= self.wave_size_lb && self.total_items <= self.wave_size_ub;
        self.rollback(); ok
    }""",
    """        let ok = self.is_stock_valid() && self.total_items >= self.wave_size_lb && self.total_items <= self.wave_size_ub;
        self.rollback(); 
        if t_start.elapsed().as_micros() > 100 {
            // println!("Slow move: {:?} took {} us", mv, t_start.elapsed().as_micros());
        }
        ok
    }"""
)

with open("src/evaluator/stock_balance.rs", "w") as f:
    f.write(text)
