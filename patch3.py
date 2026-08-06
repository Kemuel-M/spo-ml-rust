import re
with open("src/local_searchs/configurable.rs", "r") as f:
    text = f.read()

text = text.replace(
    """        for mv in moves {
            if evaluator.validate_move(&mv, data) {""",
    """        let t_start_inner = std::time::Instant::now();
        for mv in moves {
            if t_start_inner.elapsed().as_secs() > 1 { break; }
            if evaluator.validate_move(&mv, data) {"""
)

with open("src/local_searchs/configurable.rs", "w") as f:
    f.write(text)
