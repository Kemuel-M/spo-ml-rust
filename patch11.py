import re
with open("src/local_searchs/configurable.rs", "r") as f:
    text = f.read()

text = text.replace(
    "        let t_start_inner = std::time::Instant::now();\n",
    ""
)
text = text.replace(
    "            if t_start_inner.elapsed().as_secs() > 1 { break; }\n",
    ""
)

with open("src/local_searchs/configurable.rs", "w") as f:
    f.write(text)
