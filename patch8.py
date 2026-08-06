import re
with open("src/evaluator/stock_balance.rs", "r") as f:
    text = f.read()

text = text.replace(
    """        self.orders = self.snapshot_orders.clone();
        self.aisles = self.snapshot_aisles.clone();
        self.total_items = self.snapshot_total_items;
        self.violations = self.snapshot_violations;
        self.violating_items = self.snapshot_violating_items.clone();""",
    """        self.orders.clone_from(&self.snapshot_orders);
        self.aisles.clone_from(&self.snapshot_aisles);
        self.total_items = self.snapshot_total_items;
        self.violations = self.snapshot_violations;
        self.violating_items.clone_from(&self.snapshot_violating_items);"""
)

with open("src/evaluator/stock_balance.rs", "w") as f:
    f.write(text)
