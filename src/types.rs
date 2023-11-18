use bitvec::prelude::*;

use crate::error::Result;
use crate::utils::*;

pub type Encoding<T> = Vec<T>;
pub type Pair<K, V> = (K, V);

pub trait EmmV {
    fn len() -> usize;
    fn encode(&self) -> Vec<u8>;
    fn decode(b: &[u8]) -> Self;
}

/// Oblivious Key-Value Stores
pub trait Okvs {
    fn encode<K: OkvsK, V: OkvsV>(&self, input: Vec<Pair<K, V>>) -> Result<Encoding<V>>;
    fn decode<V: OkvsV>(&self, encoding: &Encoding<V>, key: &impl OkvsK) -> V;
}

pub trait OkvsK {
    fn hash_to_index(&self, range: usize) -> usize;
    fn hash_to_band(&self, band_width: usize) -> BitVec;
    fn to_bytes(&self) -> Vec<u8>;
}

pub trait OkvsV: Clone {
    fn default() -> Self;
    fn is_zero(&self) -> bool;
    fn xor(&self, other: &Self) -> Self;
}

pub struct OkvsKey<const N: usize = 8>(pub [u8; N]);

impl<const N: usize> OkvsK for OkvsKey<N> {
    /// hash1(key) -> [0, range)
    fn hash_to_index(&self, range: usize) -> usize {
        let v = blake2b::<16>(&self.to_bytes());
        (u128::from_le_bytes(v) % range as u128) as usize
    }

    /// hash2(key) -> {0, 1}^band_width
    fn hash_to_band(&self, band_width: usize) -> BitVec {
        let v = hash(&self.0, (band_width + 7) / 8);
        let v = BitSlice::<_, Lsb0>::from_slice(&v);
        let v = &v[..band_width];

        let mut bits: BitVec = BitVec::new();

        for i in 0..band_width {
            bits.push(v[i])
        }
        bits
    }

    fn to_bytes(&self) -> Vec<u8> {
        self.0.into()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OkvsValue<const N: usize>(pub [u8; N]);

impl<const N: usize> OkvsV for OkvsValue<N> {
    fn default() -> Self {
        Self([0u8; N])
    }

    fn is_zero(&self) -> bool {
        for v in &self.0 {
            if *v != 0 {
                return false;
            }
        }
        true
    }

    fn xor(&self, other: &Self) -> Self {
        assert_eq!(self.0.len(), other.0.len());
        let mut result = [0u8; N];

        for (i, item) in self.0.iter().enumerate() {
            result[i] = item ^ other.0[i];
        }
        Self(result)
    }
}
