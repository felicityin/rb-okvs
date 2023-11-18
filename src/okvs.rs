use std::collections::HashMap;

use bitvec::prelude::*;

use crate::error::{Error, Result};
use crate::types::{Encoding, Okvs, OkvsK, OkvsV, Pair};
use crate::utils::*;

/// For small encoding sizes (i.e., high rate), one should try to fix small
/// choices of ϵ such as 0.03-0.05 In contrast, if one wishes for an
/// instantiation of RB-OKVS with fast encoding/decoding times, then one can
/// pick larger values of ϵ such as 0.07-0.1.
pub const EPSILON: f64 = 0.05;
pub const LAMBDA: usize = 40;

/// RB-OKVS, Oblivious Key-Value Stores
pub struct RbOkvs {
    columns:    usize,
    band_width: usize,
}

impl RbOkvs {
    pub fn new(kv_count: usize) -> RbOkvs {
        let columns = ((1.0 + EPSILON) * kv_count as f64) as usize;
        let band_width = columns * 40 / 100; // todo

        Self {
            columns,
            band_width,
        }
    }
}

impl Okvs for RbOkvs {
    fn encode<K: OkvsK, V: OkvsV>(&self, input: Vec<Pair<K, V>>) -> Result<Encoding<V>> {
        let (matrix, start_indexes, y) = self.create_sorted_matrix(input)?;
        simple_gauss::<V>(y, matrix, start_indexes, self.band_width)
    }

    fn decode<V: OkvsV>(&self, encoding: &Encoding<V>, key: &impl OkvsK) -> V {
        let start = key.hash_to_index(self.columns - self.band_width);
        let band = key.hash_to_band(self.band_width);
        inner_product(&self.create_row(start, &band), encoding)
    }
}

impl RbOkvs {
    fn create_sorted_matrix<K: OkvsK, V: OkvsV>(
        &self,
        input: Vec<Pair<K, V>>,
    ) -> Result<(Vec<BitVec>, Vec<usize>, Vec<V>)> {
        let n = input.len();
        let mut y_map = HashMap::new();
        let mut first_one_indexes: Vec<(usize, usize)> = vec![];
        let mut start_indexes: HashMap<usize, usize> = HashMap::new();
        let mut bands: HashMap<usize, BitVec> = HashMap::new();

        // Generate bands
        for (i, (key, value)) in input.into_iter().enumerate() {
            let band = key.hash_to_band(self.band_width);
            let start = key.hash_to_index(self.columns - self.band_width);

            let first_one_id = first_one_index(&band);
            if first_one_id == band.len() {
                // All elements are 0
                return Err(Error::ZeroRow(i));
            }
            let first_one_id = start + first_one_id;
            first_one_indexes.push((i, first_one_id));

            bands.insert(i, band);
            start_indexes.insert(i, start);
            y_map.insert(i, value);
        }

        first_one_indexes.sort_by(|a, b| a.1.cmp(&b.1));

        let mut matrix: Vec<BitVec> = vec![]; // n * columns
        let mut start_ids: Vec<usize> = vec![0; n];
        let mut y: Vec<V> = vec![];

        // Generate binary matrix
        for (k, (i, index)) in first_one_indexes.into_iter().enumerate() {
            start_ids[k] = index;
            let start = start_indexes.get(&i).unwrap().to_owned();
            let band = bands.get(&i).unwrap();
            matrix.push(self.create_row(start, band));
            y.push(y_map.get(&i).unwrap().to_owned());
        }

        Ok((matrix, start_ids, y))
    }

    fn create_row(&self, start: usize, band: &BitVec) -> BitVec {
        let mut bits: BitVec = BitVec::new();

        if start > 0 {
            bits.extend(bitvec![0; start]);
        }

        let band_len = band.len();
        bits.extend(band);

        if start + band_len < self.columns {
            bits.extend(bitvec![0; self.columns - band_len - start]);
        }
        bits
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::types::{OkvsKey, OkvsValue};

    #[test]
    fn test_okvs_key() {
        let key = OkvsKey([0; 8]);

        let pos = key.hash_to_index(30);
        assert!(pos < 30);

        let band = key.hash_to_band(10);
        assert_eq!(band.len(), 10);
    }

    #[test]
    fn test_create_sorted_matrix() {
        let mut pairs: Vec<Pair<OkvsKey, OkvsValue<32>>> = vec![];
        for i in 0..20 {
            pairs.push((OkvsKey([i; 8]), OkvsValue([i; 32])));
        }
        let rb_okvs = RbOkvs::new(pairs.len());
        let (matrix, start_indexes, _y_map) = rb_okvs.create_sorted_matrix(pairs).unwrap();

        for i in 0..matrix.len() {
            assert_eq!(start_indexes[i], first_one_index(&matrix[i]));
        }
    }

    #[test]
    fn test_rb_okvs() {
        let mut pairs: Vec<Pair<OkvsKey, OkvsValue<32>>> = vec![];
        for i in 0..100 {
            pairs.push((OkvsKey([i; 8]), OkvsValue([i; 32])));
        }
        let rb_okvs = RbOkvs::new(pairs.len());

        let encode = rb_okvs.encode(pairs).unwrap();

        for i in 0..100 {
            let decode = rb_okvs.decode(&encode, &OkvsKey([i; 8]));
            assert_eq!(decode, OkvsValue([i; 32]));
        }
    }
}
