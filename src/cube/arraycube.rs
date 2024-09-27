use std::{ops::Mul, str::FromStr};

use crate::cube::*;
use strum::*;

use const_for::const_for;

const CUBEDATA_LEN: usize = CUBE_AREA * NUM_SIDES;

type CubeData = [u8; CUBEDATA_LEN];

/// A Rubiks Cube representation, using a single array.
///
/// Fast for turning and low in memory usage
/// But it's clunky to use when needing insights about corners and edges.
#[derive(Clone, PartialEq, Eq, Hash, std::fmt::Debug)]
pub struct ArrayCube {
	pub data: [u8; CUBEDATA_LEN],
}

impl Default for ArrayCube {
	/// Creates a *solved* rubiks cube!
	fn default() -> Self {
		Self { data: T_BASE }
	}
}

/// Chain 2 transformations (t1 and t2) to one transformation.
/// It returns a new transformation, in which first t1 is applied, then t2.
pub const fn chain_transform(t1: CubeData, t2: CubeData) -> CubeData {
	let mut out = [0; CUBEDATA_LEN];

	const_for!(i in 0..CUBEDATA_LEN => {
		out[i] = t1[ t2[i] as usize ];
	});

	out
}

const fn is_base(t1: CubeData) -> bool {
	const_for!(i in 0..CUBEDATA_LEN => {
		if t1[i] != i as u8 { return false; }
	});

	true
}

// ==== TRANSFORMATION MATRICES =====

/*
* The transformation-"matrix".
* Let t be the transformation, s the old state and n the new state:
* n[i] = s[ t[i] ] holds true

 * The following lists are carefully constructed.
 *
 * Naming is as follows:
 * T_{Name} or T_S_{Name}
 * where S stands for (S)ymmetry
 */

// Neutral Transformation: Does nothing
const T_BASE: CubeData = [
	0, 1, 2, 3, 4, 5, 6, 7, 8, // up
	9, 10, 11, 12, 13, 14, 15, 16, 17, // down
	18, 19, 20, 21, 22, 23, 24, 25, 26, // back
	27, 28, 29, 30, 31, 32, 33, 34, 35, // front
	36, 37, 38, 39, 40, 41, 42, 43, 44, // left
	45, 46, 47, 48, 49, 50, 51, 52, 53, // right
];

const T_UP: CubeData = [
	6, 3, 0, 7, 4, 1, 8, 5, 2, // up (totally changed)
	9, 10, 11, 12, 13, 14, 15, 16, 17, // down (unchanged)
	36, 37, 38, 21, 22, 23, 24, 25, 26, // back
	45, 46, 47, 30, 31, 32, 33, 34, 35, // front
	27, 28, 29, 39, 40, 41, 42, 43, 44, // left
	18, 19, 20, 48, 49, 50, 51, 52, 53, // right
];

const T_DOWN: CubeData = [
	0, 1, 2, 3, 4, 5, 6, 7, 8, // up
	15, 12, 9, 16, 13, 10, 17, 14, 11, // down (down)
	18, 19, 20, 21, 22, 23, 51, 52, 53, // back
	27, 28, 29, 30, 31, 32, 42, 43, 44, // front
	36, 37, 38, 39, 40, 41, 24, 25, 26, // left
	45, 46, 47, 48, 49, 50, 33, 34, 35, // right
];

const T_FRONT: CubeData = [
	0, 1, 2, 3, 4, 5, 44, 41, 38, // up
	51, 48, 45, 12, 13, 14, 15, 16, 17, // down
	18, 19, 20, 21, 22, 23, 24, 25, 26, // back
	33, 30, 27, 34, 31, 28, 35, 32, 29, // front (done)
	36, 37, 9, 39, 40, 10, 42, 43, 11, // left
	6, 46, 47, 7, 49, 50, 8, 52, 53, // right
];

const T_BACK: CubeData = [
	47, 50, 53, 3, 4, 5, 6, 7, 8, // up
	9, 10, 11, 12, 13, 14, 36, 39, 42, // down
	24, 21, 18, 25, 22, 19, 26, 23, 20, // back (done)
	27, 28, 29, 30, 31, 32, 33, 34, 35, // front
	2, 37, 38, 1, 40, 41, 0, 43, 44, // left
	45, 46, 17, 48, 49, 16, 51, 52, 15, // right
];

const T_LEFT: CubeData = [
	26, 1, 2, 23, 4, 5, 20, 7, 8, // up
	27, 10, 11, 30, 13, 14, 33, 16, 17, // down
	18, 19, 15, 21, 22, 12, 24, 25, 9, // back
	0, 28, 29, 3, 31, 32, 6, 34, 35, // front
	42, 39, 36, 43, 40, 37, 44, 41, 38, // left
	45, 46, 47, 48, 49, 50, 51, 52, 53, // right
];

const T_RIGHT: CubeData = [
	0, 1, 29, 3, 4, 32, 6, 7, 35, // up
	9, 10, 24, 12, 13, 21, 15, 16, 18, // down
	8, 19, 20, 5, 22, 23, 2, 25, 26, // back
	27, 28, 11, 30, 31, 14, 33, 34, 17, // front
	36, 37, 38, 39, 40, 41, 42, 43, 44, // left
	51, 48, 45, 52, 49, 46, 53, 50, 47, // right
];

const fn generate_transformation_table() -> [[CubeData; NUM_TURNWISES]; NUM_TURNTYPES] {
	const BASE: [CubeData; NUM_SIDES] = [T_UP, T_DOWN, T_BACK, T_FRONT, T_LEFT, T_RIGHT];

	let mut out = [[T_BASE; NUM_TURNWISES]; NUM_TURNTYPES];

	const_for!(i in 0..NUM_SIDES => {
		out[i][0] = BASE[i];
		const_for!(j in 1..NUM_TURNWISES => {
			out[i][j] = chain_transform(out[i][j-1], BASE[i]);
		});
	});

	// Advanced turntypes
	out[TurnType::M as usize][0] =
		chain_transform(out[TurnType::R as usize][0], out[TurnType::L as usize][2]);
	out[TurnType::E as usize][0] =
		chain_transform(out[TurnType::D as usize][0], out[TurnType::U as usize][2]);
	out[TurnType::S as usize][0] =
		chain_transform(out[TurnType::B as usize][0], out[TurnType::F as usize][2]);

	out[TurnType::MC as usize][0] =
		chain_transform(out[TurnType::R as usize][0], out[TurnType::L as usize][0]);
	out[TurnType::EC as usize][0] =
		chain_transform(out[TurnType::D as usize][0], out[TurnType::U as usize][0]);
	out[TurnType::SC as usize][0] =
		chain_transform(out[TurnType::B as usize][0], out[TurnType::F as usize][0]);

	const_for!(i in NUM_SIDES..NUM_TURNTYPES => {
		const_for!(j in 1..NUM_TURNWISES => {
			out[i][j] = chain_transform(out[i][j-1], out[i][0]);
		});
	});

	out
}

/// All the transformation matrices for the ArrayCube
const TRANSFORM: [[CubeData; NUM_TURNWISES]; NUM_TURNTYPES] = generate_transformation_table();

/// Create a transformation matrix from the given turns
fn convert_vec_to_transformation(turns: &Vec<Turn>) -> CubeData {
	let mut out = T_BASE;

	for turn in turns {
		let t = TRANSFORM[turn.side as usize][turn.wise as usize];
		out = chain_transform(t, out);
	}

	out
}

// =========

#[derive(thiserror::Error, Debug)]
pub enum ArrayCubeFromStrError {
	#[error("The given string does not have length {}", CUBEDATA_LEN)]
	Length,
	#[error("The corner at position {0} has a invalid color combination")]
	Corner(Corner),
	#[error("The corner at position {0} has a invalid color permutation")]
	CornerOrder(Corner),
	#[error("The edge at position {0} has a invalid color combination")]
	Edge(Edge),
}

impl FromStr for ArrayCube {
	type Err = ArrayCubeFromStrError;

	fn from_str(s: &str) -> Result<Self, Self::Err> {
		if s.len() != CUBEDATA_LEN {
			return Err(ArrayCubeFromStrError::Length);
		}

		let mut cube = ArrayCube::new();

		// Parse the colors from the string
		for (i, c) in s.as_bytes().iter().enumerate() {
			cube.data[i] = (c - b'a') * CUBE_AREA as u8;
		}

		// The center pieces have a fixed index
		for i in (4..54).step_by(CUBE_AREA) {
			cube.data[i] = i as u8;
		}

		for pos in Corner::iter() {
			let (c, o) = match cube.get_corner_at_pos(pos) {
				Some(v) => v,
				None => return Err(ArrayCubeFromStrError::Corner(pos)),
			};

			// The 3 indices to write to
			let indices: [usize; 3] = corner_to_indices(pos).into();
			// The 3 indices to write there
			let cols: [usize; 3] = corner_to_indices(c).into();

			for (i, idx) in indices.into_iter().enumerate() {
				let colidx = (3 - o + i) % 3;
				if cube.data[idx] as usize / CUBE_AREA == cols[colidx] / CUBE_AREA {
					cube.data[idx] = cols[colidx] as u8;
				} else {
					return Err(ArrayCubeFromStrError::CornerOrder(pos));
				}
			}
		}

		for pos in Edge::iter() {
			let (e, o) = match cube.get_edge_at_pos(pos) {
				Some(v) => v,
				None => return Err(ArrayCubeFromStrError::Edge(pos)),
			};

			// The 2 indices to write to
			let indices: [usize; 2] = edge_to_indices(pos).into();
			// The 2 indices to write there
			let cols: [usize; 2] = edge_to_indices(e).into();

			for (i, idx) in indices.into_iter().enumerate() {
				cube.data[idx] = cols[(o + i) % 2] as u8;
			}
		}

		Ok(cube)
	}
}

impl From<ArrayCube> for String {
	fn from(val: ArrayCube) -> Self {
		// The string saves the color
		val.data
			.iter()
			.map(|c| ((c / CUBE_AREA as u8) + b'a') as char)
			.collect()
	}
}

impl RubiksCube for ArrayCube {
	fn apply_turn(&mut self, turn: Turn) {
		// Get the transformation matrix (which is easy because it's carefully sorted)
		let transform = TRANSFORM[turn.side as usize][turn.wise as usize];
		self.apply_transform(transform);
	}
}

/// Return the indices of an ArrayCube given the corner c as a position.
///
/// ```
/// use rubikscube::prelude::*;
///
/// let cube = ArrayCube::new(); // solved cube
/// let (u,f) = edge_to_indices(Edge::UF);
/// assert!(cube.color_at(u) == Side::Up);
/// assert!(cube.color_at(f) == Side::Front);
/// ```
pub const fn corner_to_indices(c: Corner) -> (usize, usize, usize) {
	// Return index of (x/y) at the given side
	const fn help(side: Side, x: usize, y: usize) -> usize {
		side as usize * CUBE_AREA + x + y * CUBE_DIM
	}

	// Get the 3 indices of the corner
	// Note: the Corner::URF means: First Up-Index, then Right-Index, then Front-Index of the corner
	match c {
		Corner::URF => (
			help(Side::Up, 2, 2),
			help(Side::Right, 0, 0),
			help(Side::Front, 2, 0),
		),
		Corner::UBR => (
			help(Side::Up, 2, 0),
			help(Side::Back, 0, 0),
			help(Side::Right, 2, 0),
		),
		Corner::DLF => (
			help(Side::Down, 0, 0),
			help(Side::Left, 2, 2),
			help(Side::Front, 0, 2),
		),
		Corner::DFR => (
			help(Side::Down, 2, 0),
			help(Side::Front, 2, 2),
			help(Side::Right, 0, 2),
		),

		Corner::ULB => (
			help(Side::Up, 0, 0),
			help(Side::Left, 0, 0),
			help(Side::Back, 2, 0),
		),
		Corner::UFL => (
			help(Side::Up, 0, 2),
			help(Side::Front, 0, 0),
			help(Side::Left, 2, 0),
		),
		Corner::DRB => (
			help(Side::Down, 2, 2),
			help(Side::Right, 2, 2),
			help(Side::Back, 0, 2),
		),
		Corner::DBL => (
			help(Side::Down, 0, 2),
			help(Side::Back, 2, 2),
			help(Side::Left, 0, 2),
		),
	}
}

/// Return the indices of an ArrayCube given the edge e as a position.
///
/// ```
/// use rubikscube::prelude::*;
///
/// let cube = ArrayCube::new(); // solved cube
/// let (u,f) = edge_to_indices(Edge::UF);
/// assert!(cube.color_at(u) == Side::Up);
/// assert!(cube.color_at(f) == Side::Front);
/// ```
pub const fn edge_to_indices(e: Edge) -> (usize, usize) {
	// Return index of (x/y) at the given side
	const fn help(side: Side, x: usize, y: usize) -> usize {
		side as usize * CUBE_AREA + x + y * CUBE_DIM
	}

	// Get the 2 indices of the edge
	// Note that e.g Edge::UF means: First the Up-Index, then the Front-Index of the Edge
	match e {
		Edge::UF => (help(Side::Up, 1, 2), help(Side::Front, 1, 0)),
		Edge::UR => (help(Side::Up, 2, 1), help(Side::Right, 1, 0)),
		Edge::UB => (help(Side::Up, 1, 0), help(Side::Back, 1, 0)),
		Edge::UL => (help(Side::Up, 0, 1), help(Side::Left, 1, 0)),

		Edge::DF => (help(Side::Down, 1, 0), help(Side::Front, 1, 2)),
		Edge::DR => (help(Side::Down, 2, 1), help(Side::Right, 1, 2)),
		Edge::DB => (help(Side::Down, 1, 2), help(Side::Back, 1, 2)),
		Edge::DL => (help(Side::Down, 0, 1), help(Side::Left, 1, 2)),

		Edge::FR => (help(Side::Front, 2, 1), help(Side::Right, 0, 1)),
		Edge::BR => (help(Side::Back, 0, 1), help(Side::Right, 2, 1)),
		Edge::BL => (help(Side::Back, 2, 1), help(Side::Left, 0, 1)),
		Edge::FL => (help(Side::Front, 0, 1), help(Side::Left, 2, 1)),
	}
}

/// When, printed, the cube is laid out in a grid.
/// This grid converts the grid coordinate to the index
/// of the cube data
#[rustfmt::skip]
pub const DISPLAY_GRID: [[usize; 4 * CUBE_DIM]; 3 * CUBE_DIM] = [
	[99, 99, 99,  0,  1,  2, 99, 99, 99, 99, 99, 99],
	[99, 99, 99,  3,  4,  5, 99, 99, 99, 99, 99, 99],
	[99, 99, 99,  6,  7,  8, 99, 99, 99, 99, 99, 99],
	[36, 37, 38, 27, 28, 29, 45, 46, 47, 18, 19, 20],
	[39, 40, 41, 30, 31, 32, 48, 49, 50, 21, 22, 23],
	[42, 43, 44, 33, 34, 35, 51, 52, 53, 24, 25, 26],
	[99, 99, 99,  9, 10, 11, 99, 99, 99, 99, 99, 99],
	[99, 99, 99, 12, 13, 14, 99, 99, 99, 99, 99, 99],
	[99, 99, 99, 15, 16, 17, 99, 99, 99, 99, 99, 99],
];

impl ArrayCube {
	/// Create a new solved cube
	pub fn new() -> Self {
		Self::default()
	}

	/// Apply a given transformation to the cube
	pub fn apply_transform(&mut self, trans: CubeData) {
		let bef = self.data;
		for i in 0..CUBEDATA_LEN {
			self.data[i] = bef[trans[i] as usize];
		}
	}

	/// Return the color at IDX
	pub fn color_at(&self, idx: usize) -> Side {
		Side::from_repr(self.data[idx] / CUBE_AREA as u8).unwrap()
	}

	/// Print the cube in the *standard output* with ANSI-colors
	pub fn print(&self) {
		for row in DISPLAY_GRID.iter() {
			for entry in row.iter() {
				if *entry < CUBEDATA_LEN {
					print!("{}▀ ", get_ansii_color(self.color_at(*entry)));
				} else {
					print!("  ");
				}
			}
			println!();
		}

		// Reset ansii color
		print!("\x1b[00m");
	}

	/// Apply the given sequence of turns.
	pub fn apply_turns(&mut self, turns: std::vec::Vec<Turn>) {
		for turn in turns {
			self.apply_turn(turn);
		}
	}

	/// Return the sym-th symmetry of the current cube
	pub fn get_symmetry(&self, sym: usize) -> ArrayCube {
		let inv = SYMMETRY_INVERSE[sym];
		let t = chain_transform(SYMMETRIES[sym], chain_transform(self.data, SYMMETRIES[inv]));

		ArrayCube { data: t }
	}

	/// Get the inverse symmetry cube of the sym-th symmetry
	pub fn get_inv_symmetry(&self, sym: usize) -> ArrayCube {
		self.get_symmetry(SYMMETRY_INVERSE[sym])
	}

	/// Returns the corner at the position and it's orientation
	/// When you turn the corner piece to it's original place, without turning the front/back, left/right side
	/// an odd number of quarters, the orientation is as follows:
	/// 0, if it's correctly in it's place
	/// 1, if it's twisted once in clockwise direction.
	/// 2, if it's twisted counterclockwise once.
	pub fn get_corner_at_pos(&self, pos: Corner) -> Option<(Corner, usize)> {
		// Get indices of the corner
		let idx: [usize; 3] = corner_to_indices(pos).into();

		// Get the color at the indices
		let cols: Vec<_> = idx.into_iter().map(|i| self.color_at(i)).collect();

		// Get the corner from the given colors
		let corner = Corner::parse_corner(&cols)?;
		let ori = match corner {
			Corner::URF | Corner::UBR | Corner::ULB | Corner::UFL => {
				cols.iter().position(|c| *c == Side::Up)?
			}
			Corner::DLF | Corner::DFR | Corner::DRB | Corner::DBL => {
				cols.iter().position(|c| *c == Side::Down)?
			}
		};

		Some((corner, ori))
	}

	/// Returns the edge at the position and it's orientation
	/// If you would put the edge piece to it's home place without turning the front/back side an
	/// odd number of quarters, the orientation is:
	/// 0, if it's correct
	/// 1, if it's flipped
	pub fn get_edge_at_pos(&self, pos: Edge) -> Option<(Edge, usize)> {
		// Get indices of the corner
		let idx: [usize; 2] = edge_to_indices(pos).into();

		// Get the color at the indices
		let cols: Vec<_> = idx.into_iter().map(|i| self.color_at(i)).collect();

		// Get the edge from the given colors
		let edge = Edge::parse_edge(&cols)?;

		// Find out the orientation
		// If the colors are in the order, it is not flipped
		let ori = match edge {
			Edge::UF | Edge::UR | Edge::UB | Edge::UL => cols[0] != Side::Up,
			Edge::DF | Edge::DR | Edge::DB | Edge::DL => cols[0] != Side::Down,
			Edge::FR | Edge::FL => cols[0] != Side::Front,
			Edge::BR | Edge::BL => cols[0] != Side::Back,
		};

		Some((edge, ori as usize))
	}

	/// Return true if the cube is solved
	pub fn is_solved(&self) -> bool {
		self.data == T_BASE
	}
}

impl From<CubeData> for ArrayCube {
	fn from(item: CubeData) -> Self {
		Self { data: item }
	}
}

impl From<Vec<Turn>> for ArrayCube {
	fn from(item: Vec<Turn>) -> Self {
		Self::from(convert_vec_to_transformation(&item))
	}
}

impl Mul for ArrayCube {
	type Output = Self;

	fn mul(self, rhs: Self) -> Self {
		Self {
			data: chain_transform(self.data, rhs.data),
		}
	}
}

// ===== Symmetry Transformations =====

const T_S_URF3: CubeData = [
	33, 30, 27, 34, 31, 28, 35, 32, 29, // up
	24, 21, 18, 25, 22, 19, 26, 23, 20, // down
	38, 41, 44, 37, 40, 43, 36, 39, 42, // back
	51, 48, 45, 52, 49, 46, 53, 50, 47, // front
	9, 10, 11, 12, 13, 14, 15, 16, 17, // left
	8, 7, 6, 5, 4, 3, 2, 1, 0, // right
];

const T_S_F2: CubeData = [
	17, 16, 15, 14, 13, 12, 11, 10, 9, // up
	8, 7, 6, 5, 4, 3, 2, 1, 0, // down
	26, 25, 24, 23, 22, 21, 20, 19, 18, // back
	35, 34, 33, 32, 31, 30, 29, 28, 27, // front
	53, 52, 51, 50, 49, 48, 47, 46, 45, // left
	44, 43, 42, 41, 40, 39, 38, 37, 36, // right
];

const T_S_U4: CubeData = [
	6, 3, 0, 7, 4, 1, 8, 5, 2, // up
	11, 14, 17, 10, 13, 16, 9, 12, 15, // down
	36, 37, 38, 39, 40, 41, 42, 43, 44, // back
	45, 46, 47, 48, 49, 50, 51, 52, 53, // front
	27, 28, 29, 30, 31, 32, 33, 34, 35, // left
	18, 19, 20, 21, 22, 23, 24, 25, 26, // right
];

const T_S_LR2: CubeData = [
	2, 1, 0, 5, 4, 3, 8, 7, 6, // up
	11, 10, 9, 14, 13, 12, 17, 16, 15, // down
	20, 19, 18, 23, 22, 21, 26, 25, 24, // back
	29, 28, 27, 32, 31, 30, 35, 34, 33, // front
	47, 46, 45, 50, 49, 48, 53, 52, 51, // left
	38, 37, 36, 41, 40, 39, 44, 43, 42, // right
];

/// Generate all transformation based from the base symmetries
const fn generate_symmetries() -> [CubeData; NUM_SYMMETRIES] {
	let mut out = [[0; CUBEDATA_LEN]; NUM_SYMMETRIES];

	const_for!(x1 in 0..3 => {
		const_for!(x2 in 0..2 => {
			const_for!(x3 in 0..4 => {
				const_for!(x4 in 0..2 => {
					let mut t = T_BASE;
					const_for!(_ in 0..x1 => {
						t = chain_transform(t, T_S_URF3);
					});
					const_for!(_ in 0..x2 => {
						t = chain_transform(t, T_S_F2);
					});
					const_for!(_ in 0..x3 => {
						t = chain_transform(t, T_S_U4);
					});
					const_for!(_ in 0..x4 => {
						t = chain_transform(t, T_S_LR2);
					});

					let idx = 16*x1 + 8*x2 + 2*x3 + x4;
					out[idx] = t;
				});
			});

		});
	});

	out
}

/// List of all symmetry transformations
const SYMMETRIES: [CubeData; NUM_SYMMETRIES] = generate_symmetries();

/// Generate the list of indices where OUT[i] is the index of the inverse of
/// SYM[i]. That is, SYM[i] * SYM[OUT[i]] = T_BASE, for SYM = SYMMETRIES
const fn generate_symmetry_inverse_list() -> [usize; NUM_SYMMETRIES] {
	let mut out = [NUM_SYMMETRIES; NUM_SYMMETRIES];
	const SYM: [CubeData; NUM_SYMMETRIES] = generate_symmetries();

	const_for!(i in 0..NUM_SYMMETRIES => {
		const_for!(j in 0.. NUM_SYMMETRIES => {
			if is_base(chain_transform(SYM[i], SYM[j])) {
				out[i] = j;
				break;
			}
		});
	});

	out
}

/// Symmetry index of 'i' is SYM[i]
const SYMMETRY_INVERSE: [usize; NUM_SYMMETRIES] = generate_symmetry_inverse_list();

#[allow(dead_code)]
/// Return the i-th symmetry of cube

// ===== Tests =====

#[cfg(test)]
mod tests {
	use std::str::FromStr;

	use crate::cube::arraycube::ArrayCube;
	use crate::cube::*;
	use arraycube::*;

	#[test]
	/// Test for basic turning and their correctness
	fn array_cube_turns1() {
		let mut cube = ArrayCube::default();
		// Little scramble
		cube.apply_turns(random_sequence(20));

		for side in TurnType::iter() {
			let turn_n = Turn {
				side,
				wise: TurnWise::Clockwise,
			};
			let turn_c = Turn {
				side,
				wise: TurnWise::CounterClockwise,
			};
			let turn2 = Turn {
				side,
				wise: TurnWise::Double,
			};

			let mut cube_n = cube.clone();
			cube_n.apply_turn(turn_n);

			let mut cube_c = cube.clone();
			cube_c.apply_turn(turn_c);

			let mut cube2 = cube.clone();
			cube2.apply_turn(turn2);

			// Check that every turnwise isn't another one
			assert_ne!(cube_n, cube2);
			assert_ne!(cube2, cube_c);
			assert_ne!(cube_n, cube_c);

			// Check that two quarters are equal to one half
			cube_n.apply_turn(turn_n);
			assert_eq!(cube_n, cube2);

			// Check that 3 quarters are equal to one quarter counterclockwise
			cube_n.apply_turn(turn_n);
			assert_eq!(cube_n, cube_c);
		}
	}

	#[test]
	/// Test for more basic turning
	fn array_cube_turns2() {
		let mut cube = ArrayCube::default();
		let bef = cube.clone();

		// This sequence should turn back to the solved cube.
		let turns = parse_turns("U2 D2 B2 F2 L2 R2 B2 F2 L2 R2 U2 D2").unwrap();
		cube.apply_turns(turns);

		assert_eq!(cube, bef);
	}

	#[test]
	/// Test for corner parsing
	fn corner_edge_checking() {
		let mut cube = ArrayCube::default();

		for edge in Edge::iter() {
			let (e, o) = cube.get_edge_at_pos(edge).unwrap();
			assert_eq!(e, edge);
			assert_eq!(o, 0);
		}

		for corner in Corner::iter() {
			let (c, o) = cube.get_corner_at_pos(corner).unwrap();
			assert_eq!(c, corner);
			assert_eq!(o, 0);
		}

		cube.apply_turn(Turn::from_str("F").unwrap());

		let mut cnt = 0;
		for edge in Edge::iter() {
			let (_e, o) = cube.get_edge_at_pos(edge).unwrap();
			cnt += o;
		}
		assert_eq!(cnt, 4);
	}

	// ===== Transformation checks =====

	/// Check whether given transformation is a permutation
	fn check_permutation(perm: CubeData) -> bool {
		let mut has_num = [false; CUBEDATA_LEN];

		for i in 0..CUBEDATA_LEN {
			let t = perm[i] as usize;
			if has_num[t] {
				return false;
			}
			has_num[t] = true;
		}

		true
	}

	#[test]
	/// Test that every transformation permutation are actually permutations
	fn permutation_test() {
		for i in 0..NUM_TURNTYPES {
			for j in 0..NUM_TURNWISES {
				assert!(check_permutation(TRANSFORM[i][j]));
			}
		}
	}

	#[test]
	/// Test that every transformation permutation are actually permutations
	fn symmetry_permutation_test() {
		for i in 0..NUM_SYMMETRIES {
			assert!(check_permutation(SYMMETRIES[i]));
		}
	}

	#[test]
	/// Test whether all symmetries have an inverse
	fn symmetry_inverse_test() {
		for i in 0..NUM_SYMMETRIES {
			let inv = SYMMETRY_INVERSE[i];
			if NUM_SYMMETRIES <= inv {
				panic!("Symmetry {} has no inverse!", i);
			}
			let t = chain_transform(SYMMETRIES[i], SYMMETRIES[inv]);
			assert!(is_base(t));
		}
	}

	#[test]
	/// Each mirror should give legal cubes afterward
	fn legal_symmetries() {
		let cube = ArrayCube::from(parse_turns("R").unwrap());
		for i in 0..NUM_SYMMETRIES {
			let c = cube.get_symmetry(i);

			for corner in Corner::iter() {
				assert!(c.get_corner_at_pos(corner).is_some());
			}
			for edge in Edge::iter() {
				assert!(c.get_edge_at_pos(edge).is_some());
			}
		}
	}

	#[test]
	/// Check the conversion between ArrayCube and Strings
	fn arraycube_string_conversion() {
		let turns = random_sequence(40);
		let mut cube = ArrayCube::new();

		for turn in turns.into_iter() {
			cube.apply_turn(turn);

			let s: String = cube.clone().into();
			match ArrayCube::from_str(&s) {
				Ok(c) => assert_eq!(c, cube),
				Err(e) => panic!("ArrayCube conversion failed: {}", e),
			}
		}
	}
}
