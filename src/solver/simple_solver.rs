use crate::solution::{ChallengeSolution, ProblemData};
use crate::heuristics::{SolverStrategy, ConstructiveAlgorithm, LocalSearchAlgorithm};
use crate::evaluator::Evaluator;

pub struct SimpleSolver {
    pub constructive: Box<dyn ConstructiveAlgorithm>,
    pub local_search: Box<dyn LocalSearchAlgorithm>,
}

impl SolverStrategy for SimpleSolver {
    fn name(&self) -> String {
        format!("SimpleSolver [{} -> {}]", self.constructive.name(), self.local_search.name())
    }

    fn solve(&self, data: &ProblemData, seed: u64) -> ChallengeSolution {
        let mut solution = self.constructive.construct(data, seed, None);
        
        // Use o Evaluator para o score inicial, pois ele faz sincronizações automáticas
        let evaluator = crate::evaluator::StockBalanceEvaluator::new(&solution, data);
        let initial_obj = evaluator.current_objective();
        println!("Initial Score: {:.4}", initial_obj);

        self.local_search.refine(&mut solution, data, seed + 1);
        solution
    }
}
