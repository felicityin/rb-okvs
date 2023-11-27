use sp_core::U256;

use crate::error::Result;
use crate::types::{Encoding, Okvs, OkvsK, OkvsV, Pair};
use crate::utils::*;

/// For small encoding sizes (i.e., high rate), one should try to fix small
/// choices of ϵ such as 0.03-0.05 In contrast, if one wishes for an
/// instantiation of RB-OKVS with fast encoding/decoding times, then one can
/// pick larger values of ϵ such as 0.07-0.1.
const EPSILON: f64 = 0.1;
const _LAMBDA: usize = 20;
const BAND_WIDTH: usize = 128; // ((LAMBDA as f64 + 15.21) / 0.2691) as usize = 130

/// RB-OKVS, Oblivious Key-Value Stores
pub struct RbOkvs {
    columns:    usize,
    band_width: usize,
}

impl RbOkvs {
    pub fn new(kv_count: usize) -> RbOkvs {
        let columns = ((1.0 + EPSILON) * kv_count as f64) as usize;

        Self {
            columns,
            band_width: if BAND_WIDTH < columns {
                BAND_WIDTH
            } else {
                columns * 80 / 100
            },
        }
    }
}

impl Okvs for RbOkvs {
    fn encode<K: OkvsK, V: OkvsV>(&self, input: Vec<Pair<K, V>>) -> Result<Encoding<V>> {
        let (matrix, start_pos, y) = self.create_sorted_matrix(input)?;
        simple_gauss::<V>(y, matrix, start_pos, self.columns)
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

        input.iter().enumerate().for_each(|(i, (k, _))| {
            start_pos[i] = (i, k.hash_to_index(self.columns - self.band_width))
        });

        radix_sort(&mut start_pos, self.columns - self.band_width - 1);

        let mut matrix: Vec<U256> = vec![U256::default(); n];
        let mut start_ids: Vec<usize> = vec![0; n];
        let mut y: Vec<V> = vec![V::default(); n];

        // Generate binary matrix
        start_pos
            .into_iter()
            .enumerate()
            .for_each(|(k, (i, start))| {
                matrix[k] = input[i].0.hash_to_band(self.band_width);
                y[k] = input[i].1.to_owned();
                start_ids[k] = start;
            });

        Ok((matrix, start_ids, y))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{OkvsKey, OkvsValue};
    extern crate test;

    #[test]
    fn test_okvs_key() {
        let key = OkvsKey([0; 8]);

        let pos = key.hash_to_index(30);
        assert!(pos < 30);

        let band = key.hash_to_band(10);
        assert!(band.bits() <= 10);
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
        for i in 0..1000 {
            pairs.push((
                OkvsKey((i as usize).to_le_bytes()),
                OkvsValue((i as u32).to_le_bytes()),
            ));
        }
        let rb_okvs = RbOkvs::new(pairs.len());

        let encode = rb_okvs.encode(pairs).unwrap();

        for i in 0..1000 {
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
                rb_okvs.columns,
            )
            .unwrap();
        });
    }

    #[bench]
    fn bench_encode(b: &mut test::Bencher) {
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
        let mut pairs: Vec<Pair<OkvsKey, OkvsValue<1>>> = vec![];
        for i in 0..100000 {
            pairs.push((
                OkvsKey((i as usize).to_le_bytes()),
                OkvsValue((i as u8).to_le_bytes()),
            ));
        }
        let rb_okvs = RbOkvs::new(pairs.len());

        let encode = rb_okvs.encode(pairs).unwrap();

        b.iter(|| {
            for i in 0..100000 {
                rb_okvs.decode(&encode, &OkvsKey((i as usize).to_le_bytes()));
            }
        });
    }
}
