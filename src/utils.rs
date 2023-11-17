use std::cmp::min;

use bitvec::prelude::*;
use blake2::{Blake2b512, Digest};

use crate::error::{Error, Result};
use crate::types::Value;

/// Martin Dietzfelbinger and Stefan Walzer. Efficient Gauss Elimination for
/// Near-Quadratic Matrices with One Short Random Block per Row, with
/// Applications. In 27th Annual European Symposium on Algorithms (ESA 2019).
/// Schloss Dagstuhl-Leibniz-Zentrum fuer Informatik, 2019.
pub fn simple_gauss(
    mut y: Vec<Value>,
    mut matrix: Vec<BitVec>,
    start_indexes: Vec<usize>,
    band_width: usize,
) -> Result<Vec<u64>> {
    let rows = matrix.len();
    assert!(rows > 0);
    assert_eq!(rows, start_indexes.len());
    let cols = matrix[0].len();

    let mut pivot: Vec<usize> = vec![usize::MAX; rows];
    for i in 0..rows {
        for j in start_indexes[i]..min(start_indexes[i] + band_width, cols) {
            if matrix[i][j] {
                pivot[i] = j;
                for k in (i + 1)..rows {
                    if start_indexes[k] <= pivot[i] && matrix[k][pivot[i]] {
                        matrix[k] = matrix[k].clone() ^ &matrix[i];
                        y[k] ^= y[i];
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
    let mut x = vec![0u64; cols]; // solution to Ax = y
    for i in (0..rows).rev() {
        x[pivot[i]] = inner_product(&matrix[i], &x) ^ y[i];
    }
    Ok(x)
}

pub fn inner_product(m: &BitVec, x: &[Value]) -> Value {
    assert_eq!(m.len(), x.len());
    let mut result = 0u64;
    for i in 0..m.len() {
        if m[i] && x[i] != 0 {
            result ^= x[i]
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

    #[test]
    fn test_gaussian() {
        let mut matrix = vec![];
        matrix.push(bitvec![1, 1, 0, 0]);
        matrix.push(bitvec![0, 1, 1, 0]);
        matrix.push(bitvec![0, 0, 1, 1]);

        let start_indexes = vec![0, 1, 2];
        let y = vec![0, 1, 2];

        let x = simple_gauss(y.clone(), matrix.clone(), start_indexes, 2).unwrap();

        assert_eq!(inner_product(&matrix[0], &x), y[0]);
        assert_eq!(inner_product(&matrix[1], &x), y[1]);
        assert_eq!(inner_product(&matrix[2], &x), y[2]);
    }

    #[test]
    fn test_inner_product() {
        let a = bitvec![1, 1, 0, 0];
        let b = vec![1, 1, 0, 0];
        assert_eq!(0, inner_product(&a, &b));
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
