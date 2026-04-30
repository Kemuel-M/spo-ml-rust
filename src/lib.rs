#[macro_export]
macro_rules! debug_println {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        println!($($arg)*);
    }
}

pub mod io;
pub mod solution;
pub mod evaluator;
pub mod solver;
pub mod heuristics;
pub mod local_searchs;
pub mod metaheuristics;
