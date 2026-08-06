import re
with open("src/metaheuristics/bpso.rs", "r") as f:
    text = f.read()

text = text.replace(
    "let t_start_const = Instant::now();",
    """println!("Particle {} starting constructive...", i);
                let t_start_const = Instant::now();"""
)

text = text.replace(
    "let t_start_ls = Instant::now();",
    """println!("Particle {} finished constructive in {:?}. starting LS...", i, t_start_const.elapsed());
                let t_start_ls = Instant::now();"""
)

text = text.replace(
    "let time_ls = t_start_ls.elapsed().as_secs_f64();",
    """let time_ls = t_start_ls.elapsed().as_secs_f64();
                println!("Particle {} finished LS in {}s", i, time_ls);"""
)

with open("src/metaheuristics/bpso.rs", "w") as f:
    f.write(text)
