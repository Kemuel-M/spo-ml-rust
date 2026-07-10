import csv
from collections import defaultdict

with open("outputs/benchmark_local_search_report3.csv", "r") as f:
    reader = csv.DictReader(f)
    stats = defaultdict(lambda: {"total_gain": 0.0, "total_gap": 0.0, "count": 0, "improved_count": 0, "time": 0.0})
    for row in reader:
        ls = row["local_search"]
        if ls == "none": continue
        
        # Filtrar o construtivo. Queremos analisar o WORST (static) ou mediano para ver quem melhorou mais
        # No arquivo velho, tem 'BEST' e 'WORST'
        if row["constr_quality"] != "WORST": continue
        
        try:
            gain = float(row["improvement"])
            gap = float(row["gap_percent"].strip("%"))
            t = float(row["time_seconds"])
            improved = 1 if row["improved"] == "true" else 0
            
            stats[ls]["total_gain"] += gain
            stats[ls]["total_gap"] += gap
            stats[ls]["time"] += t
            stats[ls]["improved_count"] += improved
            stats[ls]["count"] += 1
        except Exception:
            pass

print("Local Search    | Avg Gain | Avg Gap % | Improved % | Avg Time")
for ls, s in sorted(stats.items(), key=lambda x: x[1]["total_gain"], reverse=True):
    if s["count"] == 0: continue
    avg_gain = s["total_gain"] / s["count"]
    avg_gap = s["total_gap"] / s["count"]
    avg_time = s["time"] / s["count"]
    improved_perc = (s["improved_count"] / s["count"]) * 100
    print(f"{ls.ljust(15)} | {avg_gain:8.4f} | {avg_gap:8.2f}% | {improved_perc:9.1f}% | {avg_time:8.4f}s")
