use aes_gcm::{
    aead::{Aead, KeyInit, OsRng},
    Aes256Gcm, Key as AesKey,
};
use sha256::digest;

use crate::error::{Error, Result};
use crate::types::{EmmV, Encoding, Okvs, OkvsKey, OkvsValue, Pair};
use crate::utils::hash;

type KF = [u8; 32];
type KE = [u8; 32];
pub type EmmPair<K, V> = (K, V);
pub const H_LEN: usize = 64;

#[derive(Default)]
pub struct ClientState {
    pub kf: KF,
    pub ke: KE,
}

// Volume-Hiding Encrypted Multi-Maps
pub struct VhEmm<T: Okvs, const OKVS_K_SIZE: usize, const OKVS_V_SIZE: usize> {
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

impl<T: Okvs, const OKVS_K_SIZE: usize, const OKVS_V_SIZE: usize>
    VhEmm<T, OKVS_K_SIZE, OKVS_V_SIZE>
{
    pub fn new(okvs: T) -> Self {
        Self { okvs }
    }

    pub fn setup<V: EmmV>(
        &self,
        input: Vec<EmmPair<u64, Vec<V>>>,
    ) -> Result<(Encoding<OkvsValue<OKVS_V_SIZE>>, ClientState)> {
        let client_state = ClientState::default();
        let mut new_input: Vec<Pair<OkvsKey<OKVS_K_SIZE>, OkvsValue<OKVS_V_SIZE>>> = vec![];

        for (key, value) in input {
            let h = sha256(&client_state.kf, key); // H_LEN = h.len()
            for (j, v) in value.iter().enumerate() {
                let k = create_key::<OKVS_K_SIZE>(h.clone(), j);
                let v = encode_value::<V, OKVS_V_SIZE>(&client_state.ke, h.clone(), v);
                new_input.push((k, v));
            }
        }

        let emm = self.okvs.encode(new_input)?;
        Ok((emm, client_state))
    }

    // TODO: split it to client's and server's
    pub fn query<V: EmmV>(
        &self,
        key: u64,
        v_len: usize,
        client_state: &ClientState,
        emm: &Encoding<OkvsValue<OKVS_V_SIZE>>,
    ) -> Result<Vec<V>> {
        let h = sha256(&client_state.kf, key);

        // TODO: send h to server

        // server
        let mut x = vec![];
        for i in 0..v_len {
            let k = create_key::<OKVS_K_SIZE>(h.clone(), i);
            x.push(self.okvs.decode(emm, &k));
        }

        // TODO: send x to client

        // client
        let mut v = vec![];
        for (i, xi) in x.into_iter().enumerate() {
            let (dh, y) = decode_value::<V, OKVS_V_SIZE>(&client_state.ke, xi);
            if dh != h {
                return Err(Error::Decode(i));
            }
            v.push(y);
        }
        Ok(v)
    }
}

fn sha256(kf: &KF, key: u64) -> Vec<u8> {
    let mut arr = kf.to_vec();
    arr.extend_from_slice(&key.to_le_bytes());
    digest(arr).into_bytes()
}

fn create_key<const OKVS_K_SIZE: usize>(mut h: Vec<u8>, i: usize) -> OkvsKey<OKVS_K_SIZE> {
    h.extend_from_slice(&i.to_le_bytes());

    let k = hash(&h, OKVS_K_SIZE);

    let mut buf = [0u8; OKVS_K_SIZE];
    buf.copy_from_slice(&k);
    OkvsKey(buf)
}

fn encode_value<V: EmmV, const OKVS_V_SIZE: usize>(
    ke: &KE,
    mut h: Vec<u8>,
    v: &V,
) -> OkvsValue<OKVS_V_SIZE> {
    h.extend_from_slice(&v.encode());

    let key = AesKey::<Aes256Gcm>::from_slice(ke);
    let cipher = Aes256Gcm::new(key);
    let ciphertext = cipher.encrypt(&Default::default(), h.as_slice()).unwrap(); // TODO: randome nonce

    let mut v = [0u8; OKVS_V_SIZE];
    v.copy_from_slice(&ciphertext);
    OkvsValue(v)
}

fn decode_value<V: EmmV, const OKVS_V_SIZE: usize>(
    ke: &KE,
    v: OkvsValue<OKVS_V_SIZE>,
) -> (Vec<u8>, V) {
    let key = AesKey::<Aes256Gcm>::from_slice(ke);
    let cipher = Aes256Gcm::new(key);
    let decode = cipher.encrypt(&Default::default(), v.0.as_slice()).unwrap(); // TODO: randome nonce
    (
        decode[..H_LEN].into(),
        V::decode(&decode[H_LEN..H_LEN + V::len()]),
    )
}

#[cfg(test)]
mod test {
    use crate::okvs::RbOkvs;

    use super::*;

    #[test]
    fn test_rb_mm_vu64() {
        pub struct EmmValue(pub u64);

        impl EmmV for EmmValue {
            fn len() -> usize {
                8
            }

            fn encode(&self) -> Vec<u8> {
                self.0.to_le_bytes().into()
            }

            fn decode(b: &[u8]) -> Self {
                let mut v = [0u8; 8];
                v.copy_from_slice(b);
                Self(u64::from_le_bytes(v))
            }
        }

        let mut pairs: Vec<EmmPair<u64, Vec<EmmValue>>> = vec![];
        for i in 0..200 {
            pairs.push((i as u64, vec![EmmValue(i as u64)]));
        }
        let rb_okvs = RbOkvs::new(pairs.len());

        // 83 = 80 + EmmValue.len()
        let rb_mm = VhEmm::<RbOkvs, 8, 88>::new(rb_okvs);
        let (emm, client_state) = rb_mm.setup(pairs).unwrap();

        for i in 0..200 {
            let value: Vec<EmmValue> = rb_mm.query(i as u64, 1, &client_state, &emm).unwrap();
            assert_eq!(value[0].0, i as u64);
        }
    }

    #[test]
    fn test_rb_mm_vu32() {
        pub struct EmmValue(pub u32);

        impl EmmV for EmmValue {
            fn len() -> usize {
                4
            }

            fn encode(&self) -> Vec<u8> {
                self.0.to_le_bytes().into()
            }

            fn decode(b: &[u8]) -> Self {
                let mut v = [0u8; 4];
                v.copy_from_slice(b);
                Self(u32::from_le_bytes(v))
            }
        }

        let mut pairs: Vec<EmmPair<u64, Vec<EmmValue>>> = vec![];
        for i in 0..200 {
            pairs.push((i as u64, vec![EmmValue(i as u32)]));
        }
        let rb_okvs = RbOkvs::new(pairs.len());

        // 83 = 80 + EmmValue.len()
        let rb_mm = VhEmm::<RbOkvs, 8, 84>::new(rb_okvs);
        let (emm, client_state) = rb_mm.setup(pairs).unwrap();

        for i in 0..200 {
            let value: Vec<EmmValue> = rb_mm.query(i as u64, 1, &client_state, &emm).unwrap();
            assert_eq!(value[0].0, i as u32);
        }
    }

    #[test]
    fn test_rb_mm_vstring() {
        pub struct EmmValue(pub String);

        impl EmmV for EmmValue {
            fn len() -> usize {
                3
            }

            fn encode(&self) -> Vec<u8> {
                self.0.as_bytes().to_vec()
            }

            fn decode(b: &[u8]) -> Self {
                Self(String::from_utf8(b.to_vec()).unwrap())
            }
        }

        let mut pairs: Vec<EmmPair<u64, Vec<EmmValue>>> = vec![];
        for i in 0..200 {
            pairs.push((i as u64, vec![EmmValue(format!("{:03}", i))]));
        }
        let rb_okvs = RbOkvs::new(pairs.len());

        // 83 = 80 + EmmValue.len()
        let rb_mm = VhEmm::<RbOkvs, 8, 83>::new(rb_okvs);
        let (emm, client_state) = rb_mm.setup(pairs).unwrap();

        for i in 0..200 {
            let value: Vec<EmmValue> = rb_mm.query(i as u64, 1, &client_state, &emm).unwrap();
            assert_eq!(value[0].0, format!("{:03}", i));
        }
    }
}
