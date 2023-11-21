use std::collections::HashMap;

use sp_core::U256;

use crate::error::{Error, Result};
use crate::types::{Encoding, Okvs, OkvsK, OkvsV, Pair};
use crate::utils::*;

/// For small encoding sizes (i.e., high rate), one should try to fix small
/// choices of ϵ such as 0.03-0.05 In contrast, if one wishes for an
/// instantiation of RB-OKVS with fast encoding/decoding times, then one can
/// pick larger values of ϵ such as 0.07-0.1.
pub const EPSILON: f64 = 0.1;
pub const LAMBDA: usize = 40;

/// RB-OKVS, Oblivious Key-Value Stores
pub struct RbOkvs {
    columns:    usize,
    band_width: usize,
}

impl RbOkvs {
    pub fn new(kv_count: usize) -> RbOkvs {
        let columns = ((1.0 + EPSILON) * kv_count as f64) as usize;
        let band_width = ((LAMBDA as f64 + 15.21) / 0.2691) as usize; // 205

        Self {
            columns,
            band_width: if band_width < columns {
                band_width
            } else {
                columns * 80 / 100
            },
        }
    }
}

impl Okvs for RbOkvs {
    fn encode<K: OkvsK, V: OkvsV>(&self, input: Vec<Pair<K, V>>) -> Result<Encoding<V>> {
        let (matrix, start_pos, first_one_pos, y) = self.create_sorted_matrix(input)?;
        simple_gauss::<V>(
            y,
            matrix,
            start_pos,
            first_one_pos,
            self.band_width,
            self.columns,
        )
    }

    fn decode<V: OkvsV>(&self, encoding: &Encoding<V>, key: &impl OkvsK) -> V {
        let start = key.hash_to_index(self.columns - self.band_width);
        let band = key.hash_to_band(self.band_width);
        inner_product(&band, encoding, start)
    }
}

impl RbOkvs {
    #[allow(clippy::type_complexity)]
    fn create_sorted_matrix<K: OkvsK, V: OkvsV>(
        &self,
        input: Vec<Pair<K, V>>,
    ) -> Result<(Vec<U256>, Vec<usize>, Vec<usize>, Vec<V>)> {
        let n = input.len();
        let mut y_map = HashMap::new();
        let mut first_one_pos: Vec<(usize, usize)> = vec![];
        let mut start_pos: HashMap<usize, usize> = HashMap::new();
        let mut bands: HashMap<usize, U256> = HashMap::new();

        // Generate bands
        for (i, (key, value)) in input.into_iter().enumerate() {
            let band = key.hash_to_band(self.band_width);
            let start = key.hash_to_index(self.columns - self.band_width);

            let first_one_id = first_one_index(&band);
            if first_one_id == 256 {
                // All elements are 0
                return Err(Error::ZeroRow(i));
            }
            let first_one_id = start + first_one_id;
            first_one_pos.push((i, first_one_id));

            bands.insert(i, band);
            start_pos.insert(i, start);
            y_map.insert(i, value);
        }

        first_one_pos.sort_by(|a, b| a.1.cmp(&b.1));

        let mut matrix: Vec<U256> = vec![]; // n * columns
        let mut start_ids: Vec<usize> = vec![0; n];
        let mut first_one_ids: Vec<usize> = vec![0; n];
        let mut y: Vec<V> = vec![];

        // Generate binary matrix
        for (k, (i, first_one)) in first_one_pos.into_iter().enumerate() {
            matrix.push(bands.get(&i).unwrap().to_owned());

            y.push(y_map.get(&i).unwrap().to_owned());

            start_ids[k] = *start_pos.get(&i).unwrap();
            first_one_ids[k] = first_one;
        }

        Ok((matrix, start_ids, first_one_ids, y))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{OkvsKey, OkvsValue};
    use bitvec::prelude::*;
    extern crate test;

    #[test]
    fn test_okvs_key() {
        let key = OkvsKey([0; 8]);

        let pos = key.hash_to_index(30);
        assert!(pos < 30);

        let band = key.hash_to_band(10);
        assert_eq!(band.bits(), 10);
    }

    #[test]
    fn test_create_sorted_matrix() {
        let mut pairs: Vec<Pair<OkvsKey, OkvsValue<32>>> = vec![];
        for i in 0..20 {
            pairs.push((OkvsKey([i; 8]), OkvsValue([i; 32])));
        }
        let rb_okvs = RbOkvs::new(pairs.len());
        let res = rb_okvs.create_sorted_matrix(pairs);
        assert!(res.is_ok());
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

    #[bench]
    fn bench_create_sorted_matrix(b: &mut test::Bencher) {
        let mut pairs: Vec<Pair<OkvsKey, OkvsValue<1>>> = vec![];
        for i in 0..1000 {
            pairs.push((
                OkvsKey((i as usize).to_le_bytes()),
                OkvsValue((i as u8).to_le_bytes()),
            ));
        }
        let rb_okvs = RbOkvs::new(pairs.len());

        b.iter(|| {
            rb_okvs.create_sorted_matrix(pairs.clone()).unwrap();
        });
    }

    #[bench]
    fn bench_simple_gauss(b: &mut test::Bencher) {
        let mut pairs: Vec<Pair<OkvsKey, OkvsValue<1>>> = vec![];
        for i in 0..1000 {
            pairs.push((
                OkvsKey((i as usize).to_le_bytes()),
                OkvsValue((i as u8).to_le_bytes()),
            ));
        }
        let rb_okvs = RbOkvs::new(pairs.len());
        let (matrix, start_pos, first_one_pos, y) = rb_okvs.create_sorted_matrix(pairs).unwrap();

        b.iter(|| {
            simple_gauss::<OkvsValue<1>>(
                y.clone(),
                matrix.clone(),
                start_pos.clone(),
                first_one_pos.clone(),
                rb_okvs.band_width,
                rb_okvs.columns,
            )
            .unwrap();
        });
    }

    #[bench]
    fn bench_encode(b: &mut test::Bencher) {
        let mut pairs: Vec<Pair<OkvsKey, OkvsValue<1>>> = vec![];
        for i in 0..10000 {
            pairs.push((
                OkvsKey((i as usize).to_le_bytes()),
                OkvsValue((i as u8).to_le_bytes()),
            ));
        }
        let rb_okvs = RbOkvs::new(pairs.len());

        b.iter(|| {
            rb_okvs.encode(pairs.clone()).unwrap();
        });
    }

    #[bench]
    fn bench_decode(b: &mut test::Bencher) {
        let mut pairs: Vec<Pair<OkvsKey, OkvsValue<1>>> = vec![];
        for i in 0..1000 {
            pairs.push((
                OkvsKey((i as usize).to_le_bytes()),
                OkvsValue((i as u8).to_le_bytes()),
            ));
        }
        let rb_okvs = RbOkvs::new(pairs.len());

        let encode = rb_okvs.encode(pairs).unwrap();

        b.iter(|| {
            for i in 0..1000 {
                rb_okvs.decode(&encode, &OkvsKey((i as u8).to_le_bytes()));
            }
        });
    }

    #[bench]
    fn bench_bitarr(b: &mut test::Bencher) {
        let mut a = bitarr![0; 16384];
        let c = bitarr![1; 16384];

        b.iter(|| {
            for _ in 0..16384 {
                a ^= c;
            }
        });
    }

    #[bench]
    fn bench_bitvec(b: &mut test::Bencher) {
        let mut a = bitvec![0; 16384];
        let c = bitvec![1; 16384];

        b.iter(|| {
            for _ in 0..16384 {
                a ^= &c;
            }
        });
    }
}
