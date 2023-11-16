use std::collections::HashMap;

use bitvec::prelude::*;

use crate::error::Result;
use crate::types::{Encoding, Key, Okvs, Pair, Value};
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
        let band_width = ((LAMBDA as f64 + 19.830) / 0.2751) as usize; // todo

        Self {
            columns,
            band_width: if band_width < columns {
                band_width
            } else {
                columns * 40 / 100
            },
        }
    }
}

impl Okvs for RbOkvs {
    fn encode<K: Key>(&self, input: Vec<Pair<K>>) -> Result<Encoding> {
        let y = input.iter().map(|a| a.1).collect();
        let (matrix, start_indexes) = self.create_sorted_matrix(input);
        simple_gauss(y, matrix, start_indexes, self.band_width)
    }

    fn decode(&self, encoding: &Encoding, key: &impl Key) -> Value {
        let start = key.hash_to_position(self.columns - self.band_width);
        let value = key.hash_to_band(self.band_width);
        inner_product(&value, &encoding[start..(start + self.band_width)].into())
    }
}

impl RbOkvs {
    fn create_sorted_matrix<K: Key>(&self, input: Vec<Pair<K>>) -> (Vec<Vec<bool>>, Vec<usize>) {
        let n = input.len();
        let mut start_one_indexes: Vec<(usize, usize)> = vec![];
        let mut start_indexes: HashMap<usize, usize> = HashMap::new();
        let mut bands: HashMap<usize, Vec<bool>> = HashMap::new();

        for (i, (key, _value)) in input.into_iter().enumerate() {
            let band = key.hash_to_band(self.band_width);
            let start = key.hash_to_position(self.columns - self.band_width);

            bands.insert(i, band.clone());
            start_indexes.insert(i, start);

            let first_one_id = first_one_index(&band);
            let first_one_id = if first_one_id == band.len() {
                // All elements are 0
                self.columns
            } else {
                start + first_one_id
            };
            start_one_indexes.push((i, first_one_id));
        }

        start_one_indexes.sort_by(|a, b| a.1.cmp(&b.1));

        // TODO: reduce the storage space of the matrix to (n * band_width)
        let mut matrix = vec![vec![false; self.columns]; n]; // n * columns
        let mut start_ids: Vec<usize> = vec![0; n];

        for (k, (i, index)) in start_one_indexes.into_iter().enumerate() {
            start_ids[k] = index;
            let start = start_indexes.get(&i).unwrap().to_owned();

            for (j, v) in bands.get(&i).unwrap().iter().enumerate() {
                matrix[k][start + j] = *v;
            }
        }

        (matrix, start_ids)
    }
}

pub struct OkvsKey(pub [u8; 8]);

impl Key for OkvsKey {
    /// hash1(key) -> [0, range)
    fn hash_to_position(&self, range: usize) -> usize {
        let v = blake2b::<16>(&self.to_bytes());
        (u128::from_le_bytes(v) % range as u128) as usize
    }

    /// hash2(key) -> {0, 1}^band_width
    fn hash_to_band(&self, band_width: usize) -> Vec<bool> {
        let v = hash(&self.0, (band_width + 7) / 8);
        let v = BitSlice::<_, Lsb0>::from_slice(&v).to_bitvec();
        let v: Vec<bool> = v.into_iter().collect();
        v[0..band_width].into()
    }

    fn to_bytes(&self) -> Vec<u8> {
        self.0.into()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_okvs_key() {
        let key = OkvsKey([0; 8]);

        let pos = key.hash_to_position(30);
        assert!(pos < 30);

        let band = key.hash_to_band(10);
        assert_eq!(band.len(), 10);
    }

    #[test]
    fn test_create_sorted_matrix() {
        let mut pairs: Vec<Pair<OkvsKey>> = vec![];
        for i in 0..10 {
            pairs.push((OkvsKey([i; 8]), false));
        }
        let rb_okvs = RbOkvs::new(pairs.len());
        let (matrix, start_indexes) = rb_okvs.create_sorted_matrix(pairs);

        for i in 0..matrix.len() {
            assert_eq!(start_indexes[i], first_one_index(&matrix[i]));
        }
    }

    #[test]
    fn test_rb_okvs() {
        let mut pairs: Vec<Pair<OkvsKey>> = vec![];
        for i in 0..100 {
            pairs.push((OkvsKey([i; 8]), false));
        }
        let rb_okvs = RbOkvs::new(pairs.len());

        let encode = rb_okvs.encode(pairs).unwrap();

        for i in 0..100 {
            let decode = rb_okvs.decode(&encode, &OkvsKey([i; 8]));
            assert!(!decode);
        }
    }
}
