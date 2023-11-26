use blake2::{Blake2b512, Digest};
use sp_core::U256;

use crate::error::{Error, Result};
use crate::types::OkvsV;

/// Martin Dietzfelbinger and Stefan Walzer. Efficient Gauss Elimination for
/// Near-Quadratic Matrices with One Short Random Block per Row, with
/// Applications. In 27th Annual European Symposium on Algorithms (ESA 2019).
/// Schloss Dagstuhl-Leibniz-Zentrum fuer Informatik, 2019.
pub fn simple_gauss<V: OkvsV>(
    mut y: Vec<V>,
    mut bands: Vec<U256>,
    start_pos: Vec<usize>,
    cols: usize,
) -> Result<Vec<V>> {
    let rows = bands.len();
    assert_eq!(rows, start_pos.len());
    assert_eq!(rows, y.len());
    let mut pivot: Vec<usize> = vec![0; rows];

    for i in 0..rows {
        let y_i = y[i].clone();

        let first_one = bands[i].trailing_zeros() as usize;
        if first_one == 256 {
            return Err(Error::ZeroRow(i));
        }

        pivot[i] = first_one + start_pos[i];

        for k in (i + 1)..rows {
            if start_pos[k] > pivot[i] {
                break;
            }
            if bit(&bands[k], pivot[i] - start_pos[k]) {
                bands[k] = xor(bands[i], bands[k], first_one, pivot[i] - start_pos[k]);
                y[k].in_place_xor(&y_i);
            }
        }
    }

    // back subsitution
    let mut x = vec![V::default(); cols]; // solution to Ax = y
    for i in (0..rows).rev() {
        x[pivot[i]] = inner_product::<V>(&bands[i], &x[start_pos[i]..]).xor(&y[i]);
    }
    Ok(x)
}

const MASK: [u64; 64] = [
    0x1,
    0x2,
    0x4,
    0x8,
    0x10,
    0x20,
    0x40,
    0x80,
    0x100,
    0x200,
    0x400,
    0x800,
    0x1000,
    0x2000,
    0x4000,
    0x8000,
    0x10000,
    0x20000,
    0x40000,
    0x80000,
    0x100000,
    0x200000,
    0x400000,
    0x800000,
    0x1000000,
    0x2000000,
    0x4000000,
    0x8000000,
    0x10000000,
    0x20000000,
    0x40000000,
    0x80000000,
    0x100000000,
    0x200000000,
    0x400000000,
    0x800000000,
    0x1000000000,
    0x2000000000,
    0x4000000000,
    0x8000000000,
    0x10000000000,
    0x20000000000,
    0x40000000000,
    0x80000000000,
    0x100000000000,
    0x200000000000,
    0x400000000000,
    0x800000000000,
    0x1000000000000,
    0x2000000000000,
    0x4000000000000,
    0x8000000000000,
    0x10000000000000,
    0x20000000000000,
    0x40000000000000,
    0x80000000000000,
    0x100000000000000,
    0x200000000000000,
    0x400000000000000,
    0x800000000000000,
    0x1000000000000000,
    0x2000000000000000,
    0x4000000000000000,
    0x8000000000000000,
];

fn bit(u: &U256, index: usize) -> bool {
    if index < 64 {
        u.0[0] & MASK[index] != 0
    } else if index < 128 {
        u.0[1] & MASK[index - 64] != 0
    } else if index < 192 {
        u.0[2] & MASK[index - 128] != 0
    } else {
        u.0[3] & MASK[index - 192] != 0
    }
}

fn xor(a: U256, b: U256, start_a: usize, start_b: usize) -> U256 {
    match start_a.cmp(&start_b) {
        std::cmp::Ordering::Equal => b ^ a,
        std::cmp::Ordering::Less => {
            let diff = start_b - start_a;
            ((b >> diff) ^ a) << diff
        }
        std::cmp::Ordering::Greater => {
            let diff = start_a - start_b;
            ((b << diff) ^ a) >> diff
        }
    }
}

pub fn inner_product<V: OkvsV>(m: &U256, x: &[V]) -> V {
    let mut result = V::default();
    let bits = m.bits();

    if bits <= 64 {
        for i in 0..bits {
            if m.0[0] & MASK[i] != 0 {
                result.in_place_xor(&x[i]);
            }
        }
        return result;
    }

    for i in 0..64 {
        if m.0[0] & MASK[i] != 0 {
            result.in_place_xor(&x[i]);
        }
    }

    let x64 = &x[64..];

    if bits <= 128 {
        for i in 0..bits - 64 {
            if m.0[1] & MASK[i] != 0 {
                result.in_place_xor(&x64[i]);
            }
        }
        return result;
    }

    for i in 0..64 {
        if m.0[1] & MASK[i] != 0 {
            result.in_place_xor(&x64[i]);
        }
    }

    let x128 = &x[128..];

    if bits <= 192 {
        for i in 0..bits - 128 {
            if m.0[2] & MASK[i] != 0 {
                result.in_place_xor(&x128[i]);
            }
        }
        return result;
    }

    for i in 0..64 {
        if m.0[2] & MASK[i] != 0 {
            result.in_place_xor(&x128[i]);
        }
    }

    let x192 = &x[192..];

    for i in 0..bits - 192 {
        if m.0[3] & MASK[i] != 0 {
            result.in_place_xor(&x192[i]);
        }
    }
    result
}

pub fn blake2b<const N: usize>(data: &[u8]) -> [u8; N] {
    use blake2::digest::{Update, VariableOutput};
    use blake2::Blake2bVar;
    assert!(N <= 64);

    let mut hasher = Blake2bVar::new(N).unwrap();
    hasher.update(data);
    let mut buf = [0u8; N];
    hasher.finalize_variable(&mut buf).unwrap();
    buf
}

pub fn hash<T: AsRef<[u8]>>(data: &T, to_bytes_size: usize) -> Vec<u8> {
    let mut hasher = Blake2b512::new();

    if to_bytes_size <= 64 {
        hasher.update(data);
        let res = hasher.finalize();
        return res[0..to_bytes_size].into();
    }

    let mut result = vec![];
    let mut last_length = to_bytes_size;
    let loop_count = (to_bytes_size + 63) / 64;

    for i in 0..loop_count {
        if i == loop_count - 1 {
            result.extend(hash(data, last_length));
        } else {
            result.extend(hash(data, 64));
            last_length -= 64;
        }
    }

    result
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::types::OkvsValue;

    #[test]
    fn test_gaussian() {
        let u = U256::from(0b11);
        let matrix = vec![u, u, u];

        let start_pos = vec![0, 1, 2];

        let y = vec![
            OkvsValue([0u8; 32]),
            OkvsValue([1u8; 32]),
            OkvsValue([2u8; 32]),
        ];

        let x = simple_gauss::<OkvsValue<32>>(y.clone(), matrix.clone(), start_pos, 4).unwrap();

        assert_eq!(inner_product(&matrix[0], &x), y[0]);
        assert_eq!(inner_product(&matrix[1], &x[1..]), y[1]);
        assert_eq!(inner_product(&matrix[2], &x[2..]), y[2]);
    }

    #[test]
    fn test_bit() {
        let a = U256::from(3); // 1 1 0

        assert!(bit(&a, 0));
        assert!(bit(&a, 1));
        assert!(!bit(&a, 2));
    }

    #[test]
    fn test_inner_product() {
        let a = U256::from(3); // 1 1 0

        let b = vec![
            OkvsValue([0u8; 32]),
            OkvsValue([0u8; 32]),
            OkvsValue([0u8; 32]),
        ];
        assert!(inner_product(&a, &b).is_zero());
    }

    #[test]
    fn test_blake2b() {
        let a = [0u8; 8];
        assert_eq!(blake2b::<10>(&a).len(), 10);
        assert_eq!(blake2b::<64>(&a).len(), 64);
    }

    #[test]
    fn test_hash() {
        let a = [0u8; 8];
        assert_eq!(hash(&a, 10).len(), 10);
        assert_eq!(hash(&a, 64).len(), 64);
        assert_eq!(hash(&a, 70).len(), 70);
    }
}
