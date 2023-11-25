use std::collections::HashMap;

use sp_core::U256;

use crate::error::Result;
use crate::types::{Encoding, Okvs, OkvsK, OkvsV, Pair};
use crate::utils::*;

/// For small encoding sizes (i.e., high rate), one should try to fix small
/// choices of ϵ such as 0.03-0.05 In contrast, if one wishes for an
/// instantiation of RB-OKVS with fast encoding/decoding times, then one can
/// pick larger values of ϵ such as 0.07-0.1.
pub const EPSILON: f64 = 0.1;
pub const LAMBDA: usize = 20;

/// RB-OKVS, Oblivious Key-Value Stores
pub struct RbOkvs {
    columns:    usize,
    band_width: usize,
}

impl RbOkvs {
    pub fn new(kv_count: usize) -> RbOkvs {
        let columns = ((1.0 + EPSILON) * kv_count as f64) as usize;
        let band_width = ((LAMBDA as f64 + 15.21) / 0.2691) as usize; // 130

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
        let (matrix, start_pos, y) = self.create_sorted_matrix(input)?;
        simple_gauss::<V>(y, matrix, start_pos, self.band_width, self.columns)
    }

    fn decode<V: OkvsV>(&self, encoding: &Encoding<V>, key: &impl OkvsK) -> V {
        let start = key.hash_to_index(self.columns - self.band_width);
        let band = key.hash_to_band(self.band_width);
        inner_product(&band, &encoding[start..])
    }
}

impl RbOkvs {
    fn create_sorted_matrix<K: OkvsK, V: OkvsV>(
        &self,
        input: Vec<Pair<K, V>>,
    ) -> Result<(Vec<U256>, Vec<usize>, Vec<V>)> {
        let n = input.len();
        let mut start_pos: Vec<(usize, usize)> = vec![(0, 0); n];
        let mut bands: HashMap<usize, U256> = HashMap::new();

        // Generate bands
        for (i, (key, _value)) in input.iter().enumerate() {
            let band = key.hash_to_band(self.band_width);
            let start = key.hash_to_index(self.columns - self.band_width);

            bands.insert(i, band);
            start_pos[i] = (i, start);
        }

        start_pos.sort_by(|a, b| a.1.cmp(&b.1));

        let mut matrix: Vec<U256> = vec![U256::default(); n];
        let mut start_ids: Vec<usize> = vec![0; n];
        let mut y: Vec<V> = vec![V::default(); n];

        // Generate binary matrix
        for (k, (i, start)) in start_pos.into_iter().enumerate() {
            matrix[k] = bands.get(&i).unwrap().to_owned();
            y[k] = input[i].1.to_owned();
            start_ids[k] = start;
        }

        Ok((matrix, start_ids, y))
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
        let mut pairs: Vec<Pair<OkvsKey, OkvsValue<4>>> = vec![];
        for i in 0..10000 {
            pairs.push((
                OkvsKey((i as usize).to_le_bytes()),
                OkvsValue((i as u32).to_le_bytes()),
            ));
        }
        let rb_okvs = RbOkvs::new(pairs.len());

        let encode = rb_okvs.encode(pairs).unwrap();

        for i in 0..10000 {
            let decode = rb_okvs.decode(&encode, &OkvsKey((i as usize).to_le_bytes()));
            assert_eq!(decode, OkvsValue((i as u32).to_le_bytes()));
        }
    }

    #[bench]
    fn bench_create_sorted_matrix(b: &mut test::Bencher) {
        let mut pairs: Vec<Pair<OkvsKey, OkvsValue<1>>> = vec![];
        for i in 0..100000 {
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
        for i in 0..100000 {
            pairs.push((
                OkvsKey((i as usize).to_le_bytes()),
                OkvsValue((i as u8).to_le_bytes()),
            ));
        }
        let rb_okvs = RbOkvs::new(pairs.len());
        let (matrix, start_pos, y) = rb_okvs.create_sorted_matrix(pairs).unwrap();

        b.iter(|| {
            simple_gauss::<OkvsValue<1>>(
                y.clone(),
                matrix.clone(),
                start_pos.clone(),
                rb_okvs.band_width,
                rb_okvs.columns,
            )
            .unwrap();
        });
    }

    #[bench]
    fn bench_encode(b: &mut test::Bencher) {
        // 1000000 777ms
        let mut pairs: Vec<Pair<OkvsKey, OkvsValue<1>>> = vec![];
        for i in 0..1000000 {
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
        // 1000000 774ms
        let mut pairs: Vec<Pair<OkvsKey, OkvsValue<1>>> = vec![];
        for i in 0..1000000 {
            pairs.push((
                OkvsKey((i as usize).to_le_bytes()),
                OkvsValue((i as u8).to_le_bytes()),
            ));
        }
        let rb_okvs = RbOkvs::new(pairs.len());

        let encode = rb_okvs.encode(pairs).unwrap();

        b.iter(|| {
            for i in 0..1000000 {
                rb_okvs.decode(&encode, &OkvsKey((i as usize).to_le_bytes()));
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
