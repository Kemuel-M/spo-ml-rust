use crate::solution::{ChallengeSolution, ProblemData, DenseItem};
use anyhow::{Context, Result};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use fixedbitset::FixedBitSet;

pub fn read_input(file_path: &str, deterministic: bool) -> Result<ProblemData> {
    let file = File::open(file_path)
    .with_context(|| format!("Failed to open input file: {}", file_path))?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // Ler cabeçalho: nOrders nItems nAisles
    let header = lines.next().context("Empty input file")??;
    let parts: Vec<&str> = header.split_whitespace().collect();
    let n_orders: usize = parts[0].parse()?;
    let n_items: usize = parts[1].parse()?;
    let n_aisles: usize = parts[2].parse()?;

    // Ler pedidos
    let mut orders = Vec::with_capacity(n_orders);
    for _ in 0..n_orders {
        let line = lines.next().context("Unexpected end of file reading orders")??;
        orders.push(parse_item_quantity_pair(&line)?);
    }

    // Ler corredores
    let mut aisles = Vec::with_capacity(n_aisles);
    for _ in 0..n_aisles {
        let line = lines.next().context("Unexpected end of file reading aisles")??;
        aisles.push(parse_item_quantity_pair(&line)?);
    }

    // Ler LB e UB
    let bounds_line = lines.next().context("Missing bounds line")??;
    let bounds: Vec<&str> = bounds_line.split_whitespace().collect();
    let wave_size_lb: u32 = bounds[0].parse()?;
    let wave_size_ub: u32 = bounds[1].parse()?;

    // --- Pré-processamento de Performance ---
    use crate::solution::DenseItem;
    
    let dense_orders: Vec<Vec<DenseItem>> = orders.iter()
        .map(|m| {
            let mut v: Vec<DenseItem> = m.iter().map(|(&id, &qty)| DenseItem { id, qty }).collect();
            if deterministic {
                v.sort_unstable_by_key(|it| it.id);
            }
            v
        })
        .collect();
    
    let dense_aisles: Vec<Vec<DenseItem>> = aisles.iter()
        .map(|m| {
            let mut v: Vec<DenseItem> = m.iter().map(|(&id, &qty)| DenseItem { id, qty }).collect();
            if deterministic {
                v.sort_unstable_by_key(|it| it.id);
            }
            v
        })
        .collect();

    let order_total_items: Vec<u32> = orders.iter()
        .map(|m| m.values().sum())
        .collect();

    let mut item_to_aisles = vec![Vec::new(); n_items];
    for (a_idx, aisle_items) in aisles.iter().enumerate() {
        for &it_id in aisle_items.keys() {
            if it_id < n_items {
                item_to_aisles[it_id].push(a_idx);
            }
        }
    }

    let mut orders_sorted_by_size: Vec<usize> = (0..n_orders).collect();
    orders_sorted_by_size.sort_unstable_by(|&a, &b| {
        order_total_items[b].cmp(&order_total_items[a])
    });

    // Matriz Densa de Estoque
    let mut stock_matrix = vec![0u32; n_aisles * n_items];
    for (a_idx, aisle_items) in aisles.iter().enumerate() {
        for (&it_id, &qty) in aisle_items {
            if it_id < n_items {
                stock_matrix[a_idx * n_items + it_id] = qty;
            }
        }
    }

    // Bitsets de Localização de Itens
    use fixedbitset::FixedBitSet;
    let mut item_locations_bits = vec![FixedBitSet::with_capacity(n_aisles); n_items];
    for (a_idx, aisle_items) in aisles.iter().enumerate() {
        for &it_id in aisle_items.keys() {
            if it_id < n_items {
                item_locations_bits[it_id].insert(a_idx);
            }
        }
    }

    use std::sync::Arc;
    let mut order_required_aisles = Vec::with_capacity(n_orders);
    for o_idx in 0..n_orders {
        let req = compute_order_req(o_idx, n_items, &dense_orders[o_idx], &stock_matrix, &item_locations_bits, deterministic);
        order_required_aisles.push(req);
    }

    let mut aisle_to_orders_req = vec![Vec::new(); n_aisles];
    for (o_idx, req) in order_required_aisles.iter().enumerate() {
        for &a in req {
            aisle_to_orders_req[a].push(o_idx);
        }
    }

    let mut order_initial_aisles_count = Vec::with_capacity(n_orders);
    for req in &order_required_aisles {
        order_initial_aisles_count.push(req.len());
    }
    let all_order_indices: Vec<usize> = (0..n_orders).collect();

    Ok(ProblemData {
        orders,
        aisles,
        n_items,
        wave_size_lb,
        wave_size_ub,
        dense_orders: Arc::new(dense_orders),
        dense_aisles: Arc::new(dense_aisles),
        item_to_aisles: Arc::new(item_to_aisles),
        order_total_items: Arc::new(order_total_items),
        orders_sorted_by_size,
        stock_matrix,
        item_locations_bits,
        order_required_aisles: Arc::new(order_required_aisles),
        aisle_to_orders_req: Arc::new(aisle_to_orders_req),
        order_initial_aisles_count,
        all_order_indices,
    })
}

fn compute_order_req(
    _o_idx: usize,
    n_items: usize,
    order_items: &[DenseItem],
    stock_matrix: &[u32],
    item_locations_bits: &[FixedBitSet],
    deterministic: bool
) -> Vec<usize> {
    let mut remaining: HashMap<usize, u32> = order_items.iter().map(|it| (it.id, it.qty)).collect();
    let mut selected = Vec::new();
    
    while !remaining.is_empty() {
        let mut best_aisle = None;
        let mut max_cov = 0;
        
        let mut candidate_aisles = HashSet::new();
        for &it_id in remaining.keys() {
            for a_idx in item_locations_bits[it_id].ones() {
                if !selected.contains(&a_idx) {
                    candidate_aisles.insert(a_idx);
                }
            }
        }

        let candidates_iter: Vec<usize> = if deterministic {
            let mut v: Vec<usize> = candidate_aisles.into_iter().collect();
            v.sort_unstable();
            v
        } else {
            candidate_aisles.into_iter().collect()
        };

        for a_idx in candidates_iter {
            let mut cov = 0;
            for (&it_id, &dem) in &remaining {
                cov += std::cmp::min(dem, stock_matrix[a_idx * n_items + it_id]);
            }
            if cov > max_cov {
                max_cov = cov;
                best_aisle = Some(a_idx);
            }
        }

        if let Some(a_idx) = best_aisle {
            selected.push(a_idx);
            let mut to_remove = Vec::new();
            for (&it_id, dem) in remaining.iter_mut() {
                let stk = stock_matrix[a_idx * n_items + it_id];
                if stk >= *dem {
                    to_remove.push(it_id);
                } else {
                    *dem -= stk;
                }
            }
            for it in to_remove {
                remaining.remove(&it);
            }
        } else {
            break;
        }
    }
    selected
}

fn parse_item_quantity_pair(line: &str) -> Result<HashMap<usize, u32>> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    let count: usize = parts[0].parse()?;
    let mut map = HashMap::with_capacity(count);

    // O formato é: count item1 qty1 item2 qty2 ...
    for k in 0..count {
        let item_idx: usize = parts[1 + 2 * k].parse()?;
        let qty: u32 = parts[1 + 2 * k + 1].parse()?;
        map.insert(item_idx, qty);
    }
    Ok(map)
}

pub fn write_output(solution: &ChallengeSolution, output_path: &str) -> Result<()> {
    let file = File::create(output_path)
    .with_context(|| format!("Failed to create output file: {}", output_path))?;
    let mut writer = BufWriter::new(file);

    // Escreve número de pedidos
    writeln!(writer, "{}", solution.orders.count_ones(..))?;
    // Escreve cada pedido
    for order in solution.orders.ones() {
        writeln!(writer, "{}", order)?;
    }

    // Escreve número de corredores
    writeln!(writer, "{}", solution.aisles.count_ones(..))?;
    // Escreve cada corredor
    for aisle in solution.aisles.ones() {
        writeln!(writer, "{}", aisle)?;
    }

    println!("Output written to {}", output_path);
    Ok(())
}
