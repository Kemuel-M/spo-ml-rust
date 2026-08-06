use spo_ml_rust::io;
use spo_ml_rust::heuristics::ConstructiveAlgorithm;
use spo_ml_rust::heuristics::super_hybrid::SuperHybrid;
use std::time::Instant;

fn main() {
    let data = io::read_input("datasets/x/instance_0010.txt", false).unwrap();
    let sh = SuperHybrid::new(0.05);
    let start = Instant::now();
    for i in 0..10 {
        let s = Instant::now();
        let sol = sh.construct(&data, i as u64);
        println!("Iter {}, score = {:.4}, time = {:?}", i, sol.score, s.elapsed());
    }
    println!("Time: {:?}", start.elapsed());
}
