import re
with open("src/metaheuristics/bpso.rs", "r") as f:
    text = f.read()

text = text.replace(
    "let init_solution = guided_construct(data, &p, &sh, seed + i as u64);",
    """println!("Starting guided_construct for particle {}", i);
                let init_solution = guided_construct(data, &p, &sh, seed + i as u64);
                println!("Finished guided_construct for particle {}", i);"""
)

with open("src/metaheuristics/bpso.rs", "w") as f:
    f.write(text)
