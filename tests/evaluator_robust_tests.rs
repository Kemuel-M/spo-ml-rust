use spo_ml_rust::solution::{ChallengeSolution, ProblemData, DenseItem};
use spo_ml_rust::evaluator::{StockBalanceEvaluator, Evaluator, Move};
use std::collections::HashMap;
use std::sync::Arc;
use fixedbitset::FixedBitSet;

/// Helper para criar um ProblemData fake controlado para testes
fn create_mock_data() -> ProblemData {
    let mut orders = Vec::new();
    let mut aisles = Vec::new();

    // Pedido 0: Precisa de 10 do Item A (id 0) e 5 do Item B (id 1)
    let mut o0 = HashMap::new(); o0.insert(0, 10); o0.insert(1, 5);
    orders.push(o0);

    // Pedido 1: Precisa de 10 do Item C (id 2)
    let mut o1 = HashMap::new(); o1.insert(2, 10);
    orders.push(o1);

    // Corredor 0: Tem 20 do Item A (id 0)
    let mut a0 = HashMap::new(); a0.insert(0, 20);
    aisles.push(a0);

    // Corredor 1: Tem 10 do Item B (id 1)
    let mut a1 = HashMap::new(); a1.insert(1, 10);
    aisles.push(a1);

    // Corredor 2: Tem apenas 5 do Item C (id 2) -> INSUFICIENTE para Pedido 1 sozinho
    let mut a2 = HashMap::new(); a2.insert(2, 5);
    aisles.push(a2);

    // Corredor 3: Tem mais 10 do Item C (id 2)
    let mut a3 = HashMap::new(); a3.insert(2, 10);
    aisles.push(a3);

    let n_items = 3;
    let n_aisles = aisles.len();

    // Transformar em dados densos
    let dense_orders: Vec<Vec<DenseItem>> = orders.iter().map(|m| 
        m.iter().map(|(&id, &qty)| DenseItem { id, qty }).collect()
    ).collect();

    let dense_aisles: Vec<Vec<DenseItem>> = aisles.iter().map(|m| 
        m.iter().map(|(&id, &qty)| DenseItem { id, qty }).collect()
    ).collect();

    let mut item_to_aisles = vec![vec![]; n_items];
    for (a_idx, m) in aisles.iter().enumerate() {
        for &item_id in m.keys() {
            if item_id < n_items {
                item_to_aisles[item_id].push(a_idx);
            }
        }
    }

    let order_total_items = vec![15, 10];
    let orders_sorted_by_size = vec![0, 1];

    let mut stock_matrix = vec![0u32; n_aisles * n_items];
    for (a_idx, m) in aisles.iter().enumerate() {
        for (&it_id, &qty) in m {
            if it_id < n_items {
                stock_matrix[a_idx * n_items + it_id] = qty;
            }
        }
    }

    let mut item_locations_bits = vec![FixedBitSet::with_capacity(n_aisles); n_items];
    for (a_idx, m) in aisles.iter().enumerate() {
        for &it_id in m.keys() {
            if it_id < n_items {
                item_locations_bits[it_id].insert(a_idx);
            }
        }
    }

    // Para o teste, simplificamos o order_required_aisles
    let order_required_aisles = vec![
        vec![0, 1], // Pedido 0 precisa de Corredores 0 e 1
        vec![3],    // Pedido 1 precisa de Corredor 3 (que tem 10, enquanto 2 tem apenas 5)
    ];

    ProblemData {
        orders,
        aisles,
        n_items,
        wave_size_lb: 5,
        wave_size_ub: 50,
        dense_orders: Arc::new(dense_orders),
        dense_aisles: Arc::new(dense_aisles),
        item_to_aisles: Arc::new(item_to_aisles),
        order_total_items: Arc::new(order_total_items),
        orders_sorted_by_size,
        stock_matrix,
        item_locations_bits,
        order_required_aisles: Arc::new(order_required_aisles),
    }
}

#[test]
fn test_evaluator_aggressive_flow() {
    let data = create_mock_data();
    let solution = ChallengeSolution::new(data.orders.len(), data.aisles.len());
    
    // Inicia vazio
    let mut eval = StockBalanceEvaluator::new(&solution, &data);
    assert_eq!(eval.current_objective(), 0.0);
    assert!(eval.is_stock_valid());

    // 1. ADICIONAR PEDIDO 0 (15 itens)
    eval.try_apply_order_add(0, &data);
    assert!(eval.is_stock_valid());
    assert_eq!(eval.current_total_items(), 15);
    assert!(eval.aisles.contains(0));
    assert!(eval.aisles.contains(1));
    // Score: 15 / 2 = 7.5
    assert!((eval.current_objective() - 7.5).abs() < 1e-6);
    eval.commit();

    // 2. TESTAR ROLLBACK
    let old_obj = eval.current_objective();
    eval.try_apply_order_add(1, &data);
    assert_eq!(eval.current_total_items(), 25);
    eval.rollback();
    assert_eq!(eval.current_total_items(), 15);
    assert_eq!(eval.current_objective(), old_obj);

    // 3. TESTAR VIABILIDADE COMPLEXA
    eval.try_apply_order_add(1, &data);
    assert!(eval.is_stock_valid());
    assert!(eval.aisles.contains(3));
    
    // Itens: 15 + 10 = 25. Corredores: 0, 1 e 3 (total 3 corredores).
    // Score: 25 / 3 = 8.3333
    assert!((eval.current_objective() - 8.333333).abs() < 1e-4);
    eval.commit();

    // 4. TESTAR MINIMIZAÇÃO
    eval.try_apply_order_remove(1, &data);
    assert!(!eval.aisles.contains(3));
    assert_eq!(eval.current_total_items(), 15);
    assert!((eval.current_objective() - 7.5).abs() < 1e-6);
}

#[test]
fn test_evaluator_constraints_and_penalties() {
    let mut data = create_mock_data();
    data.wave_size_lb = 20; 
    
    let solution = ChallengeSolution::new(data.orders.len(), data.aisles.len());
    let mut eval = StockBalanceEvaluator::new(&solution, &data);
    
    eval.try_apply_order_add(0, &data);
    assert_eq!(eval.current_objective(), 0.0);
    
    data.wave_size_lb = 5;
    data.wave_size_ub = 10;
    let mut eval2 = StockBalanceEvaluator::new(&solution, &data);
    eval2.try_apply_order_add(0, &data);
    assert_eq!(eval2.current_objective(), 0.0);
}

#[test]
fn test_validate_move_integrity() {
    let data = create_mock_data();
    let solution = ChallengeSolution::new(data.orders.len(), data.aisles.len());
    let mut eval = StockBalanceEvaluator::new(&solution, &data);
    
    let mv = Move::OrderInsertion(0);
    let is_valid = eval.validate_move(&mv, &data);
    
    assert!(is_valid);
    assert_eq!(eval.current_total_items(), 0);
    assert_eq!(eval.orders.count_ones(..), 0, "Estado de pedidos deveria estar vazio após rollback");
}
