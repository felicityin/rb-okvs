use aes_gcm::{
    aead::{Aead, KeyInit, OsRng},
    Aes256Gcm, Key as AesKey,
};
use sha256::digest;

use crate::error::Result;
use crate::okvs::{Encoding, Key, Okvs, Pair, Value};
use crate::utils::hash;

type KF = [u8; 32];
type KE = [u8; 32];
pub const VALUE_SIZE: usize = 1;

#[derive(Default)]
pub struct ClientState {
    pub kf: KF,
    pub ke: KE,
}

// Volume-Hiding Encrypted Multi-Maps
pub struct VhEmm<T: Okvs> {
    okvs: T,
}

impl ClientState {
    pub fn new_random() -> Self {
        Self {
            kf: Aes256Gcm::generate_key(OsRng).into(),
            ke: Aes256Gcm::generate_key(OsRng).into(),
        }
    }
}

impl<T: Okvs> VhEmm<T> {
    pub fn new(okvs: T) -> Self {
        Self { okvs }
    }

    pub fn setup(&self, input: Vec<Pair>) -> Result<(Encoding, ClientState)> {
        let client_state = ClientState::default();
        let mut new_input: Vec<Pair> = vec![];

        for (key, value) in input.into_iter() {
            let h = sha256(&client_state.kf, &key);
            // TODO: make it general
            for j in 0..VALUE_SIZE {
                let k = create_key(concat(h.clone(), j));
                let v = create_value(&client_state.ke, concat(h.clone(), value));
                new_input.push((k, v));
            }
        }

        let emm = self.okvs.encode(new_input)?;
        Ok((emm, client_state))
    }

    // TODO: split it to client's and server's
    pub fn query(&self, key: Key, client_state: &ClientState, emm: &Encoding) -> Value {
        let h = sha256(&client_state.kf, &key);

        // TODO: send h to server

        // server
        let mut x = vec![];
        // TODO: make it general
        for i in 0..VALUE_SIZE {
            let k = create_key(concat(h.clone(), i));
            x.push(self.okvs.decode(emm, &k));
        }

        // TODO: send x to client

        // client
        let mut v = vec![];
        // TODO: make it general
        for i in x {
            let y = create_value(&client_state.ke, i.to_string());
            v.push(y);
        }
        v[0]
    }
}

fn sha256(kf: &KF, key: &Key) -> String {
    let mut arr = kf.to_vec();
    arr.extend_from_slice(key);
    digest(arr)
}

fn concat<T: std::fmt::Display>(mut a: String, b: T) -> String {
    a.push_str(&b.to_string());
    a
}

// TODO: make it general
fn create_key(s: String) -> Key {
    let k = hash(&s.as_bytes(), 8);
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&k);
    buf
}

// TODO: make it general
fn create_value(ke: &KE, e: String) -> Value {
    let key = AesKey::<Aes256Gcm>::from_slice(ke);
    let cipher = Aes256Gcm::new(key);
    let ciphertext = cipher.encrypt(&Default::default(), e.as_bytes()).unwrap(); // TODO: randome nonce
    (ciphertext[0] % 2) != 0
}

#[cfg(test)]
mod test {
    use crate::okvs::RbOkvs;

    use super::*;

    #[test]
    fn test_rb_mm() {
        let mut pairs: Vec<Pair> = vec![];
        for i in 0..200 {
            pairs.push(([i; 8], false));
        }
        let rb_okvs = RbOkvs::new(pairs.len());

        let rb_mm = VhEmm::new(rb_okvs);
        let (emm, client_state) = rb_mm.setup(pairs).unwrap();

        for i in 0..200 {
            let value = rb_mm.query([i; 8], &client_state, &emm);
            assert!(!value);
        }
    }
}
