use std::collections::HashMap;
use std::sync::Arc;
use fixedbitset::FixedBitSet;

#[derive(Debug, Clone)]
pub struct DenseItem {
    pub id: usize,
    pub qty: u32,
}

#[derive(Debug)]
pub struct ProblemData {
    pub orders: Vec<HashMap<usize, u32>>,
    pub aisles: Vec<HashMap<usize, u32>>,
    pub n_items: usize,
    pub wave_size_lb: u32,
    pub wave_size_ub: u32,

    // Cache de performance compartilhado (Arc para clones O(1))
    pub dense_orders: Arc<Vec<Vec<DenseItem>>>,
    pub dense_aisles: Arc<Vec<Vec<DenseItem>>>,
    pub item_to_aisles: Arc<Vec<Vec<usize>>>,
    pub order_total_items: Arc<Vec<u32>>,
    pub orders_sorted_by_size: Vec<usize>,

    // Estruturas de Matriz Densa para Otimização Global
    pub stock_matrix: Vec<u32>,                // Acesso rápido: [aisle_idx * n_items + item_id]
    pub item_locations_bits: Vec<FixedBitSet>, // Acesso rápido: item_id -> bitset de corredores
    pub order_required_aisles: Arc<Vec<Vec<usize>>>,
}

#[derive(Debug, Clone)]
pub struct ChallengeSolution {
    pub orders: FixedBitSet,
    pub aisles: FixedBitSet,
    pub score: f64,
    pub feasible: bool,
    pub metadata: HashMap<String, String>,
}

impl ChallengeSolution {
    pub fn new(n_orders: usize, n_aisles: usize) -> Self {
        Self {
            orders: FixedBitSet::with_capacity(n_orders),
            aisles: FixedBitSet::with_capacity(n_aisles),
            score: 0.0,
            feasible: false,
            metadata: HashMap::new(),
        }
    }

    /// Retorna o bitset da dimensão solicitada
    pub fn bits(&self, dim: crate::local_searchs::SearchDimension) -> &FixedBitSet {
        match dim {
            crate::local_searchs::SearchDimension::Orders => &self.orders,
            crate::local_searchs::SearchDimension::Aisles => &self.aisles,
        }
    }

    /// Retorna o bitset mutável da dimensão solicitada
    pub fn bits_mut(&mut self, dim: crate::local_searchs::SearchDimension) -> &mut FixedBitSet {
        match dim {
            crate::local_searchs::SearchDimension::Orders => &mut self.orders,
            crate::local_searchs::SearchDimension::Aisles => &mut self.aisles,
        }
    }
}
