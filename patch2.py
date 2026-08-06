import re
with open("src/evaluator/core_stock.rs", "r") as f:
    text = f.read()

text = text.replace(
    """    fn validate_move(&mut self, mv: &Move, data: &ProblemData) -> bool {
        // --- LOOKAHEAD RESTRICTION REMOVED ---
        
        match *mv {""",
    """    fn validate_move(&mut self, mv: &Move, data: &ProblemData) -> bool {
        // --- RESTORED LOOKAHEAD RESTRICTION ---
        let mut mv_order_score = 0.0;
        let mut mv_aisle_score = 0.0;
        match *mv {
            Move::OrderSwap(o, i) | Move::OrderInsertion(o, i) => {
                mv_order_score = data.order_scores[i] - data.order_scores[o];
            },
            Move::OrderRemoval(o) => {
                mv_order_score = -data.order_scores[o];
            },
            Move::AisleRemoval(a) => {
                mv_aisle_score = -data.aisle_scores[a];
            },
            Move::AisleSwap(o, i) => {
                mv_aisle_score = data.aisle_scores[i] - data.aisle_scores[o];
            }
        }
        
        let total_score = mv_order_score + mv_aisle_score;
        if total_score < 0.0 {
            // Early reject via lookahead score
            return false; 
        }
        
        match *mv {"""
)
with open("src/evaluator/core_stock.rs", "w") as f:
    f.write(text)
