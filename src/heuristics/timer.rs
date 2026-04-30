use crate::solution::{ChallengeSolution, ProblemData};
use crate::heuristics::{ConstructiveAlgorithm, LocalSearchAlgorithm, SolverStrategy};
use std::time::Instant;

/// Wrapper para cronometrar Algoritmos Construtivos
pub struct TimedConstructive {
    inner: Box<dyn ConstructiveAlgorithm>,
}

impl TimedConstructive {
    pub fn new(inner: Box<dyn ConstructiveAlgorithm>) -> Self {
        Self { inner }
    }
}

impl ConstructiveAlgorithm for TimedConstructive {
    fn name(&self) -> String {
        self.inner.name()
    }

    fn with_alpha(&self, alpha: f64) -> Option<Box<dyn ConstructiveAlgorithm>> {
        self.inner.with_alpha(alpha).map(|c| Box::new(TimedConstructive::new(c)) as Box<dyn ConstructiveAlgorithm>)
    }

    fn construct(&self, data: &ProblemData, seed: u64) -> ChallengeSolution {
        let start = Instant::now();
        let solution = self.inner.construct(data, seed);
        let duration = start.elapsed();
        
        println!("  [Timer] Construction ({}) took {:.4}s", self.inner.name(), duration.as_secs_f64());
        solution
    }
}

/// Wrapper para cronometrar Algoritmos de Busca Local
pub struct TimedLocalSearch {
    inner: Box<dyn LocalSearchAlgorithm>,
}

impl TimedLocalSearch {
    pub fn new(inner: Box<dyn LocalSearchAlgorithm>) -> Self {
        Self { inner }
    }
}

impl LocalSearchAlgorithm for TimedLocalSearch {
    fn name(&self) -> String {
        self.inner.name()
    }

    fn refine(&self, solution: &mut ChallengeSolution, data: &ProblemData, seed: u64) -> bool {
        let start = Instant::now();
        let improved = self.inner.refine(solution, data, seed);
        let duration = start.elapsed();
        
        println!("  [Timer] Local Search ({}) took {:.4}s (Improved: {})", 
            self.inner.name(), 
            duration.as_secs_f64(), 
            improved
        );
        improved
    }
}

/// Wrapper para cronometrar Estratégias de Solução Completas
pub struct TimedSolver {
    inner: Box<dyn SolverStrategy>,
}

impl TimedSolver {
    pub fn new(inner: Box<dyn SolverStrategy>) -> Self {
        Self { inner }
    }
}

impl SolverStrategy for TimedSolver {
    fn name(&self) -> String {
        self.inner.name()
    }

    fn solve(&self, data: &ProblemData, seed: u64) -> ChallengeSolution {
        let start = Instant::now();
        let solution = self.inner.solve(data, seed);
        let duration = start.elapsed();
        println!("  [Timer] Total Solver ({}) took {:.4}s", self.inner.name(), duration.as_secs_f64());
        solution
    }
}
