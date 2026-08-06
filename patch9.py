import re
with open("src/evaluator/stock_balance.rs", "r") as f:
    text = f.read()

text = text.replace(
    """        if t_start.elapsed().as_micros() > 100 {
            println!("Slow move: {:?} took {} us", mv, t_start.elapsed().as_micros());
        }""",
    ""
)

with open("src/evaluator/stock_balance.rs", "w") as f:
    f.write(text)
