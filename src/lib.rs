//! This is a simple example how to turn the cube.
//!
//! ```
//! use rubikscube::prelude::*;
//!
//! let mut cube = CubieCube::new();
//!
//! let turns1 = parse_turns("U2 D2 B2 F2 L2 R2").unwrap();
//! let turns2 = parse_turns("M2 E2 S2").unwrap();
//!
//! cube.apply_turns(turns1);
//! cube.apply_turns(turns2);
//!
//! assert!(cube.is_solved());
//! ```
//!
//! M, E and S are advanced turns and are combinations of U/D, B/F and L/R.

pub mod cube;
mod math;
pub mod solve;

pub mod prelude {
	pub use crate::cube::{arraycube::*, cubiecube::*, turn::*, *};
	pub use crate::solve::kociemba::{KociembaData, Solver};
}
