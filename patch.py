import re
with open("src/metaheuristics/bpso.rs", "r") as f:
    text = f.read()

text = text.replace(
    "let stats: (usize, f64, usize, usize, usize, usize, usize) = particles.par_iter_mut().map(|p| {",
    "let stats: (usize, f64, usize, usize, usize, usize, usize, f64, f64) = particles.par_iter_mut().map(|p| {"
)

text = text.replace(
    "let seed_construct = p.rng.next_u64();",
    "let seed_construct = p.rng.next_u64();\n                let t_start_const = Instant::now();"
)

text = text.replace(
    "let mut sol = self.guided_construct(data, &p.scores_buffer, p.dimension, seed_construct, &mut p.indices_buffer);",
    "let mut sol = self.guided_construct(data, &p.scores_buffer, p.dimension, seed_construct, &mut p.indices_buffer);\n                let time_const = t_start_const.elapsed().as_secs_f64();"
)

text = text.replace(
    "let mut ls_hit = 0;\n                let mut ls_attempt = 0;",
    "let mut ls_hit = 0;\n                let mut ls_attempt = 0;\n                let mut time_ls = 0.0;"
)

text = text.replace(
    "let seed_ls = p.rng.next_u64();\n                    self.local_search.refine(&mut sol, data, seed_ls);",
    "let seed_ls = p.rng.next_u64();\n                    let t_start_ls = Instant::now();\n                    self.local_search.refine(&mut sol, data, seed_ls);\n                    time_ls = t_start_ls.elapsed().as_secs_f64();"
)

text = text.replace(
    "(improved, v_sum / n as f64, ls_attempt, ls_hit, v_sat_count, h_dist_count, n)\n            }).reduce(|| (0, 0.0, 0, 0, 0, 0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3, a.4 + b.4, a.5 + b.5, a.6 + b.6));",
    "(improved, v_sum / n as f64, ls_attempt, ls_hit, v_sat_count, h_dist_count, n, time_const, time_ls)\n            }).reduce(|| (0, 0.0, 0, 0, 0, 0, 0, 0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3, a.4 + b.4, a.5 + b.5, a.6 + b.6, a.7 + b.7, a.8 + b.8));"
)

text = text.replace(
    "let total_bits = stats.6;",
    "let total_bits = stats.6;\n            let total_time_const = stats.7;\n            let total_time_ls = stats.8;"
)

text = text.replace(
    "println!(\"  {:<6} | {:<10.4} | {:<8.4} | {:>4.1}% | {:<4} | {:<4.2} | {:<4.2} | {:<5.2} | {:>4.1}% | {:>4.1}% | {:>4.1}% | {:>3}s\", \n                         iter, gbest_f, avg_fitness, imp_percent, stagnation_count, w, turbulence_chance, avg_v, ls_hit_rate, avg_sat_percent, avg_h_dist_percent, start.elapsed().as_secs());",
    "info!(\"Iter {:<3} | Best: {:.4} | Imp: {:>4.1}% | Stag: {} | LSHit: {:>4.1}% | T.Const: {:.1}s | T.LS: {:.1}s\", \n                         iter, gbest_f, imp_percent, stagnation_count, ls_hit_rate, total_time_const, total_time_ls);"
)

with open("src/metaheuristics/bpso.rs", "w") as f:
    f.write(text)
