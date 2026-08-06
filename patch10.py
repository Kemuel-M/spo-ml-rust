import re
with open("src/evaluator/stock_balance.rs", "r") as f:
    text = f.read()

text = text.replace(
    "let t_start = std::time::Instant::now();",
    ""
)

with open("src/evaluator/stock_balance.rs", "w") as f:
    f.write(text)
