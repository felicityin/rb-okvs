use sp_core::U256;

use crate::error::Result;
use crate::utils::*;

pub type Encoding<T> = Vec<T>;
pub type Pair<K, V> = (K, V);

pub trait EmmK {
    fn to_bytes(&self) -> Vec<u8>;
}

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
    fn hash_to_band(&self, band_width: usize) -> U256;
    fn to_bytes(&self) -> Vec<u8>;
}

pub trait OkvsV: Clone {
    fn default() -> Self;
    fn is_zero(&self) -> bool;
    fn xor(&self, other: &Self) -> Self;
    fn in_place_xor(&mut self, other: &Self);
}

#[derive(Clone)]
pub struct OkvsKey<const N: usize = 8>(pub [u8; N]);

impl<const N: usize> OkvsK for OkvsKey<N> {
    /// hash1(key) -> [0, range)
    fn hash_to_index(&self, range: usize) -> usize {
        let v = blake2b::<8>(&self.to_bytes());
        usize::from_le_bytes(v) % range
    }

    /// hash2(key) -> {0, 1}^band_width
    fn hash_to_band(&self, band_width: usize) -> U256 {
        let mut v = hash(&self.0, band_width / 8);
        v[0] |= 1;
        U256::from_little_endian(&v)
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
        let mut result = [0u8; N];
        for (i, item) in self.0.iter().enumerate() {
            result[i] = item ^ other.0[i];
        }
        Self(result)
    }

    fn in_place_xor(&mut self, other: &Self) {
        for i in 0..self.0.len() {
            self.0[i] ^= other.0[i];
        }
    }
}
