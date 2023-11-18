use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Row {0} is 0")]
    ZeroRow(usize),

    #[error("Decode error: {0}")]
    Decode(usize),
}
