pub mod arraycube;
pub mod cubiecube;
pub mod turn;

use strum::EnumCount;
use turn::*;

/// The dimension of the cube
pub const CUBE_DIM: usize = 3;

/// The number of facelets per side
pub const CUBE_AREA: usize = CUBE_DIM * CUBE_DIM;

/// The number of sides of a cube
pub const NUM_SIDES: usize = 6;

// TODO: Change this to an enum
#[derive(Eq, PartialEq, PartialOrd, Copy, Clone, strum::EnumCount, strum::FromRepr)]
#[repr(u8)]
pub enum Side {
	// It must only contain unit fields!
	Up,
	Down,
	Back,
	Front,
	Left,
	Right,
}

/// Returns the ANSI-colorcode for the given side.
pub fn get_ansii_color(side: Side) -> &'static str {
	match side {
		Side::Up => "\x1b[00m",    // White
		Side::Down => "\x1b[93m",  // Yellow
		Side::Back => "\x1b[32m",  // Green
		Side::Front => "\x1b[34m", // Blue
		Side::Left => "\x1b[31m",  // Red
		Side::Right => "\x1b[33m", // Orange
	}
}

// ===== Edge Piece =====

/// All the different position names for an Edge
#[derive(Clone, Copy, Default, PartialEq, Eq, strum::EnumIter, strum::EnumCount, strum::Display, Debug)]
#[repr(usize)]
#[rustfmt::skip]
pub enum Edge {
	#[default]
	UF, UR, UB, UL, // up edges
	DF, DR, DB, DL, // down edges
	FR, BR, BL, FL, // ud-slice (middle edges)
}

pub const NUM_EDGES: usize = Edge::COUNT;

#[derive(Clone, Copy, PartialEq, Eq, strum::EnumIter, strum::EnumCount, Debug)]
#[repr(usize)]
pub enum EdgeOrientation {
	Normal,
	Flipped,
}

impl Edge {
	/// Parse the edge from the given colors.
	/// Returns None if there exist no Edge with the given colors.
	pub fn parse_edge(col: &[Side]) -> Option<Self> {
		#[cfg(debug_assertions)]
		assert!(col.len() == 2);

		// Create a hash out of the color
		let hash = {
			let mut m: usize = 0;
			for c in col {
				m |= 1 << *c as usize;
			}
			m
		};

		// Each hash matches a different edge.
		let res = match hash {
			// RL_FB_DU
			0b_00_10_01 => Self::UF,
			0b_10_00_01 => Self::UR,
			0b_00_01_01 => Self::UB,
			0b_01_00_01 => Self::UL,

			0b_00_10_10 => Self::DF,
			0b_10_00_10 => Self::DR,
			0b_00_01_10 => Self::DB,
			0b_01_00_10 => Self::DL,

			0b_10_10_00 => Self::FR,
			0b_01_10_00 => Self::FL,
			0b_01_01_00 => Self::BL,
			0b_10_01_00 => Self::BR,
			_ => return None,
		};

		Some(res)
	}
}

// ===== Corner Piece =====

/// A corner piece
/// Note that the name is carefully sorted!
#[derive(
	Clone, Copy, Default, PartialEq, Eq, Debug, strum::EnumIter, strum::EnumString, strum::EnumCount, strum::Display
)]
#[allow(clippy::upper_case_acronyms)]
#[repr(usize)]
#[rustfmt::skip]
pub enum Corner {
	#[default]
	URF, UBR, DLF, DFR, // DON'T CHANGE THE ORDER OF THE LETTERS!
	ULB, UFL, DRB, DBL,
}

pub const NUM_CORNERS: usize = Corner::COUNT;

#[derive(
	Clone, Copy, PartialEq, Eq, Debug, strum::EnumIter, strum::EnumString, strum::EnumCount,
)]
#[allow(clippy::upper_case_acronyms)]
#[repr(usize)]
pub enum CornerOrientation {
	Normal,
	Clockwisetwist,
	AntiClockwisetwist,
}

impl Corner {
	/// Parse the corner from the given colors.
	/// If no corner with the given colors exist, the function returns None.
	pub fn parse_corner(col: &[Side]) -> Option<Self> {
		#[cfg(debug_assertions)]
		assert!(col.len() == 3);

		// Create a hash
		let hash = {
			let mut m: usize = 0;
			for c in col {
				m |= 1 << *c as usize;
			}
			m
		};

		// Match the hash
		let res = match hash {
			// RL_FB_DU
			0b_01_10_01 => Self::UFL,
			0b_10_10_01 => Self::URF,
			0b_01_01_01 => Self::ULB,
			0b_10_01_01 => Self::UBR,
			0b_01_10_10 => Self::DLF,
			0b_10_10_10 => Self::DFR,
			0b_01_01_10 => Self::DBL,
			0b_10_01_10 => Self::DRB,
			_ => return None,
		};

		Some(res)
	}
}

/// It contains all the different types a cube configuration
/// could illegal.
/// From it, you are able to know how to fix the cube.
#[derive(thiserror::Error, Debug)]
pub enum CubeError {
	#[error("The orientation-parity of the corners are off by +{0}")]
	CornerOrientation(usize),
	#[error("The orientation-parity of the edges are off by 1")]
	EdgeOrientation,
	#[error("The number of swaps needed is odd")]
	Permutation,
	#[error("Not all cubies are present on the cube")]
	Cubies,
}

/// The RubiksCube trait.
pub trait RubiksCube {
	fn apply_turn(&mut self, turn: Turn);
}

#[cfg(test)]
mod tests {
	use super::{arraycube::*, cubiecube::*, turn::*, *};
	use std::{error::Error, str::FromStr};

	#[test]
	/// Check that all advanced turntypes function as expected
	fn check_advanced_turns() -> Result<(), Box<dyn Error>> {
		let buildup = vec![
			(Turn::from_str("M")?, parse_turns("R L'")?),
			(Turn::from_str("E")?, parse_turns("D U'")?),
			(Turn::from_str("S")?, parse_turns("B F'")?),
			(Turn::from_str("MC")?, parse_turns("R L")?),
			(Turn::from_str("EC")?, parse_turns("D U")?),
			(Turn::from_str("SC")?, parse_turns("B F")?),
		];

		for (turn, combo) in buildup.iter() {
			let mut cubie = ArrayCube::new();
			let mut cubie2 = ArrayCube::new();

			cubie.apply_turn(*turn);
			for turn in combo.iter() {
				cubie2.apply_turn(*turn);
			}

			if cubie != cubie2 {
				panic!("Turn {}: CubieCube is not correct!", turn);
			}
		}

		for (turn, combo) in buildup.iter() {
			let mut cubie = CubieCube::new();
			let mut cubie2 = CubieCube::new();

			cubie.apply_turn(*turn);
			for turn in combo.iter() {
				cubie2.apply_turn(*turn);
			}

			if cubie != cubie2 {
				panic!("Turn {}: CubieCube is not correct!", turn);
			}
		}

		Ok(())
	}
}
