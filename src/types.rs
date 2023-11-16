use crate::error::Result;

pub type Value = bool;
pub type Encoding = Vec<bool>;
pub type Pair<T> = (T, Value);

/// Oblivious Key-Value Stores
pub trait Okvs {
    fn encode<K: Key>(&self, input: Vec<Pair<K>>) -> Result<Encoding>;
    fn decode(&self, encoding: &Encoding, key: &impl Key) -> Value;
}

pub trait Key {
    fn hash_to_position(&self, range: usize) -> usize;
    fn hash_to_band(&self, band_width: usize) -> Vec<bool>;
    fn to_bytes(&self) -> Vec<u8>;
}
