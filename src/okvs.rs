use std::collections::HashMap;

use bitvec::prelude::*;

use crate::error::Result;
use crate::utils::*;

/// For small encoding sizes (i.e., high rate), one should try to fix small
/// choices of ϵ such as 0.03-0.05 In contrast, if one wishes for an
/// instantiation of RB-OKVS with fast encoding/decoding times, then one can
/// pick larger values of ϵ such as 0.07-0.1.
pub const EPSILON: f64 = 0.05;
pub const LAMBDA: usize = 40;

pub type Key = [u8; 8];
pub type Value = bool;
pub type Encoding = Vec<bool>;
pub type Pair = (Key, Value);

/// Oblivious Key-Value Stores
pub trait Okvs {
    fn encode(&self, input: Vec<Pair>) -> Result<Encoding>;
    fn decode(&self, encoding: Encoding, key: &Key) -> Value;
}

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
    fn encode(&self, input: Vec<Pair>) -> Result<Encoding> {
        let y = input.iter().map(|a| a.1).collect();
        let (matrix, start_indexes) = self.create_sorted_matrix(input);
        simple_gauss(y, matrix, start_indexes, self.band_width)
    }

    fn decode(&self, encoding: Encoding, key: &Key) -> Value {
        let start = self.hash_to_position(key);
        let value = self.hash_to_band(key);
        inner_product(&value, &encoding[start..(start + self.band_width)].into())
    }
}

impl RbOkvs {
    fn create_sorted_matrix(&self, input: Vec<Pair>) -> (Vec<Vec<bool>>, Vec<usize>) {
        let n = input.len();
        let mut start_one_indexes: Vec<(usize, usize)> = vec![];
        let mut start_indexes: HashMap<usize, usize> = HashMap::new();
        let mut bands: HashMap<usize, Vec<bool>> = HashMap::new();

        for (i, (key, _value)) in input.into_iter().enumerate() {
            let band = self.hash_to_band(&key);
            let start = self.hash_to_position(&key);

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

    /// hash1(key) -> [0, columns - band_width)
    fn hash_to_position(&self, key: &Key) -> usize {
        let range = self.columns - self.band_width;
        let v = blake2b::<16>(key);
        (u128::from_le_bytes(v) % range as u128) as usize
    }

    /// hash2(key) -> {0, 1}^band_width
    fn hash_to_band(&self, key: &Key) -> Vec<bool> {
        let v = hash(key, (self.band_width + 7) / 8);
        let v = BitSlice::<_, Lsb0>::from_slice(&v).to_bitvec();
        let v: Vec<bool> = v.into_iter().collect();
        v[0..self.band_width].into()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_hash_to_position() {
        let rb_okvs = RbOkvs::new(30);

        let pos = rb_okvs.hash_to_position(&[0; 8]);
        assert!(pos < 30);

        let pos = rb_okvs.hash_to_position(&[21; 8]);
        assert!(pos < 30);
    }

    #[test]
    fn test_hash_to_band() {
        let rb_okvs = RbOkvs::new(3);
        let band = rb_okvs.hash_to_band(&[0; 8]);
        assert_eq!(band.len(), rb_okvs.band_width);
    }

    #[test]
    fn test_create_sorted_matrix() {
        let mut pairs: Vec<Pair> = vec![];
        for i in 0..10 {
            pairs.push(([i; 8], false));
        }
        let rb_okvs = RbOkvs::new(pairs.len());
        let (matrix, start_indexes) = rb_okvs.create_sorted_matrix(pairs);

        for i in 0..matrix.len() {
            assert_eq!(start_indexes[i], first_one_index(&matrix[i]));
        }
    }

    #[test]
    fn test_rb_okvs() {
        let mut pairs: Vec<Pair> = vec![];
        for i in 0..100 {
            pairs.push(([i; 8], false));
        }
        let rb_okvs = RbOkvs::new(pairs.len());

        let encode = rb_okvs.encode(pairs).unwrap();

        for i in 0..100 {
            let decode = rb_okvs.decode(encode.clone(), &[i; 8]);
            assert!(!decode);
        }
    }
}
