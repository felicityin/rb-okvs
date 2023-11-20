use bitvec::prelude::*;
use blake2::{Blake2b512, Digest};

use crate::error::{Error, Result};
use crate::types::OkvsV;

/// Martin Dietzfelbinger and Stefan Walzer. Efficient Gauss Elimination for
/// Near-Quadratic Matrices with One Short Random Block per Row, with
/// Applications. In 27th Annual European Symposium on Algorithms (ESA 2019).
/// Schloss Dagstuhl-Leibniz-Zentrum fuer Informatik, 2019.
pub fn simple_gauss<V: OkvsV>(
    mut y: Vec<V>,
    mut bands: Vec<BitVec>,
    start_pos: Vec<usize>,
    first_one_pos: Vec<usize>,
    band_width: usize,
    cols: usize,
) -> Result<Vec<V>> {
    let rows = bands.len();
    assert_eq!(rows, start_pos.len());
    assert_eq!(rows, y.len());
    let mut pivot: Vec<usize> = vec![usize::MAX; rows];

    for i in 0..rows {
        let y_i = y[i].clone();

        for j in first_one_pos[i]..start_pos[i] + band_width {
            if bands[i][j - start_pos[i]] {
                pivot[i] = j;
                for k in (i + 1)..rows {
                    if first_one_pos[k] <= pivot[i] && bands[k][pivot[i] - start_pos[k]] {
                        bands[k] = bxor(
                            &bands[i][j - start_pos[i] ..],
                            &bands[k],
                            pivot[i] - start_pos[k],
                        );
                        y[k].in_place_xor(&y_i);
                    }
                }
                break;
            }
        }

        if pivot[i] == usize::MAX {
            // row i is 0
            return Err(Error::ZeroRow(i));
        }
    }

    // back subsitution
    let mut x = vec![V::default(); cols]; // solution to Ax = y
    for i in (0..rows).rev() {
        x[pivot[i]] = inner_product::<V>(&bands[i], &x, start_pos[i]).xor(&y[i]);
    }
    Ok(x)
}

fn bxor(a: &BitSlice, b: &BitVec, start_b: usize) -> BitVec {
    let mut c = bitvec![];

    if start_b > 0 {
        c.extend(&b[..start_b]);
    }

    let mut i = 0;
    let mut j = start_b;

    while i < a.len() && j < b.len() {
        c.push(a[i] ^ b[j]);
        i += 1;
        j += 1;
    }

    if j < b.len() {
        c.extend(&b[j..]);
    }

    if i < a.len() {
        c.extend(&a[i..]);
    }
    c
}

pub fn inner_product<V: OkvsV>(m: &BitSlice, x: &[V], start: usize) -> V {
    let mut result = V::default();
    for i in 0..m.len() {
        if m[i] {
            result.in_place_xor(&x[start + i]);
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

pub fn first_one_index(bits: &BitVec) -> usize {
    for (i, v) in bits.iter().enumerate() {
        if v == true {
            return i;
        }
    }
    bits.len()
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::types::OkvsValue;

    #[test]
    fn test_gaussian() {
        let mut matrix = vec![];
        matrix.push(bitvec![1, 1]);
        matrix.push(bitvec![1, 1]);
        matrix.push(bitvec![1, 1]);

        let start_pos = vec![0, 1, 2];
        let first_one_pos = vec![0, 1, 2];

        let y = vec![
            OkvsValue([0u8; 32]),
            OkvsValue([1u8; 32]),
            OkvsValue([2u8; 32]),
        ];

        let x = simple_gauss::<OkvsValue<32>>(
            y.clone(),
            matrix.clone(),
            start_pos,
            first_one_pos,
            2,
            4,
        )
        .unwrap();

        assert_eq!(inner_product(&matrix[0], &x, 0), y[0]);
        assert_eq!(inner_product(&matrix[1], &x, 1), y[1]);
        assert_eq!(inner_product(&matrix[2], &x, 2), y[2]);
    }

    #[test]
    fn test_inner_product() {
        let a = bitvec![1, 1, 0];
        let b = vec![
            OkvsValue([0u8; 32]),
            OkvsValue([0u8; 32]),
            OkvsValue([0u8; 32]),
        ];
        assert!(inner_product(&a, &b, 0).is_zero());
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

    #[test]
    fn test_first_one_index() {
        let a = bitvec![0, 1];
        assert_eq!(first_one_index(&a), 1);
    }
}
