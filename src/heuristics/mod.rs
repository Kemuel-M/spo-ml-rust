// Exporta os módulos filhos
pub mod aisle_centric_static;
pub mod aisle_centric_adaptive;
pub mod aisle_centric_random;
pub mod static_greedy;
pub mod adaptive_greedy;
pub mod random_greedy;
pub mod hybrid_random;
pub mod super_hybrid;
pub mod utils;
pub mod timer;

use crate::solution::{ProblemData, ChallengeSolution};

// --- Interfaces Principais ---

/// Interface para Algoritmos Construtivos (Etapa 1).
/// Objetivo: Criar uma solução válida do zero.
pub trait ConstructiveAlgorithm: Send + Sync {
    fn construct(&self, data: &ProblemData, seed: u64) -> ChallengeSolution;
    fn name(&self) -> String;
    fn with_alpha(&self, _alpha: f64) -> Option<Box<dyn ConstructiveAlgorithm>> {
        None
    }
}

/// Interface para Busca Local (Etapa 2).
/// Objetivo: Receber uma solução e tentar melhorá-la.
/// Retorna true se houve melhoria.
pub trait LocalSearchAlgorithm: Send + Sync {
    fn refine(&self, solution: &mut ChallengeSolution, data: &ProblemData, seed: u64) -> bool;
    fn name(&self) -> String;
}

/// Interface Geral para Solvers (Engloba Etapa 1, 2 e 3).
/// Pode ser uma simples construtiva, ou uma metaheurística complexa.
pub trait SolverStrategy: Send + Sync {
    fn solve(&self, data: &ProblemData, seed: u64) -> ChallengeSolution;
    fn name(&self) -> String;
}

// --- Implementação Genérica ---

// Permite que qualquer Algoritmo Construtivo seja usado diretamente como um SolverStrategy
// (Isso mantém a compatibilidade com a Etapa 1)
impl<T: ConstructiveAlgorithm> SolverStrategy for T {
    fn solve(&self, data: &ProblemData, seed: u64) -> ChallengeSolution {
        self.construct(data, seed)
    }

    fn name(&self) -> String {
        self.name()
    }
}
