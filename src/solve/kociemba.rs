/*
* This algorithm is the algorithm Herbert Kociemba published on
* <https://kociemba.org/cube.htm>
*
* It has some changes and generalisations though.
*/

use serde::{Deserialize, Serialize};
use std::io::{Read, Write};

use rayon::prelude::*;
use strum::IntoEnumIterator;

use crate::cube::cubiecube::*;
use crate::cube::NUM_SYMMETRIES;
use crate::cube::{
	turn::{parse_turns, Turn},
	Edge, RubiksCube, NUM_EDGES,
};

use crate::math::*;

/// v[coord][i] is the coordinate when applying move i on coord
type Movetable = Vec<Vec<u16>>;

/// v[coord][i] is the coordinate of the i-th symmetry of coord
type Symtable = Vec<Vec<u16>>;

/// v[symcoord][i] = (dst, sym) results by applying the move i on the symcoordinate "symcoord"
/// where dst: Is the resulting sym-coordinate
/// where sym: Is the sym-th symmetry of the raw coordinate of dst
type SymMovetable = Vec<Vec<(u16, u8)>>;

/// A mapper vector where v[i] is the raw coordinate of the
/// representant cube of symmetry class i
type SymToRawTable = Vec<u32>;

/// A mapper vector where v[coord] is the symmetry class coord belongs to
type RawToSymTable = Vec<(u16, u8)>;

/// A state which contains all i where S[i] * A S[i]^-1 = A
type SymState = Vec<u32>;

const RLSLICE_EDGES: [Edge; 4] = [Edge::UF, Edge::DF, Edge::DB, Edge::UB];
const FBSLICE_EDGES: [Edge; 4] = [Edge::UR, Edge::DR, Edge::DL, Edge::UL];
const UDSLICE_EDGES: [Edge; 4] = [Edge::FR, Edge::BR, Edge::BL, Edge::FL];

// Helper list to map the index of the edge in an edge-slice
const EDGE_SLICE_INDEX: [usize; NUM_EDGES] = [0, 0, 3, 3, 1, 1, 2, 2, 0, 1, 2, 3];

// Cosets sizes

const SLICE: usize = 495; // 12 choose 4
const FLIP_UDSLICE: usize = EDGE_ORI * SLICE;
const SYM_FLIP_UDSLICE: usize = 64430;

const EDGE8_PERM: usize = 40320; // 8!
const SYM_CORNER_PERM: usize = 2768;
const SLICE_SORTED: usize = SLICE * 24; // 24 = 4!

// ===== Compressed Heuristic table =====

/// Heuristic struct
/// It is a compressed Vec<u8> which uses 2 bits per entry
#[derive(serde::Serialize, serde::Deserialize, Clone, PartialEq, Eq)]
struct Heuristics {
	data: Vec<u8>,
}

impl Heuristics {
	pub fn get(&self, i: usize) -> Option<u8> {
		let x = self.data.get(i >> 2)?;
		let offset = (i & 0b11) << 1;
		let res = (x >> offset) & 0b11;
		Some(res)
	}
}

impl From<Vec<u8>> for Heuristics {
	fn from(val: Vec<u8>) -> Self {
		let data = (0..val.len())
			.into_par_iter()
			.step_by(4)
			.map(|i| {
				let t = std::cmp::min(i + 4, val.len());

				(i..t)
					.map(|j| {
						let off = (j - i) << 1;
						(val[j] % 3) << off
					})
					.sum()
			})
			.collect();
		Self { data }
	}
}

// =====

#[derive(Clone, Debug)]
struct CoordPhase1 {
	// Coordinates
	corner_ori: usize,
	edge_ori: usize,
	udslice_sorted: usize,
	// Coords needed for for phase 2
	corner_perm: usize,
	fbslice_sorted: usize,
	rlslice_sorted: usize,
}

impl CoordPhase1 {
	/// Return the phase1 coordinate extracted from the cube
	pub fn new(cube: &CubieCube) -> Self {
		let corner_ori = cube.get_corner_orientation_coord();
		let edge_ori = cube.get_edge_orientation_coord();
		let corner_perm = cube.get_corner_perm_coord();

		let udslice_sorted = get_udslice_sorted_coord(cube);
		let fbslice_sorted = get_fbslice_sorted_coord(cube);
		let rlslice_sorted = get_rlslice_sorted_coord(cube);

		Self {
			corner_ori,
			edge_ori,
			udslice_sorted,
			corner_perm,
			fbslice_sorted,
			rlslice_sorted,
		}
	}
}

#[derive(Serialize, Deserialize)]
struct DataPhase1 {
	turns: Vec<Turn>,

	// Movetables
	corner_ori_move: Movetable,
	edge_ori_move: Movetable,
	slice_move: Movetable,
	corner_perm_move: Movetable,

	heuristic: Heuristics,

	cornersym: Symtable,

	into_edge8_perm: Vec<Vec<u16>>,
	rawtosym: RawToSymTable,
}

impl DataPhase1 {
	/// Get all the tables needed for phase 1
	pub fn new(turns: Vec<Turn>) -> Self {
		let corner_ori_move = create_movetable(
			CORNER_ORI,
			&turns,
			get_corner_ori_coord,
			cube_from_corner_ori,
		);

		let edge_ori_move =
			create_movetable(EDGE_ORI, &turns, get_edge_ori_coord, cube_from_edge_ori);

		let slice_move = create_movetable(
			SLICE_SORTED,
			&turns,
			get_udslice_sorted_coord,
			cube_from_udslice_sorted,
		);

		let corner_perm_move = create_movetable(
			CORNER_PERM,
			&turns,
			get_corner_perm_coord,
			cube_from_corner_perm,
		);

		let flipudslice_toraw = create_symtoraw(
			FLIP_UDSLICE,
			SYM_FLIP_UDSLICE,
			get_flip_udslice_coord,
			cube_from_flip_udslice,
		);

		let flipudslice_symstate = gen_symstate(
			&flipudslice_toraw,
			get_flip_udslice_coord,
			cube_from_flip_udslice,
		);
		let flipudslice_symmove = create_sym_movetable(
			&flipudslice_toraw,
			&turns,
			get_flip_udslice_coord,
			cube_from_flip_udslice,
		);
		let cornersym = create_symtable(CORNER_ORI, 16, get_corner_ori_coord, cube_from_corner_ori);
		let heuristic = gen_heuristics(
			&corner_ori_move,
			&flipudslice_symmove,
			&cornersym,
			&flipudslice_symstate,
		);

		let into_edge8_perm = get_into_edge8perm();

		let rawtosym =
			create_rawtosym(FLIP_UDSLICE, get_flip_udslice_coord, cube_from_flip_udslice);

		Self {
			turns: turns.clone(),
			corner_ori_move,
			edge_ori_move,
			slice_move,
			corner_perm_move,
			heuristic,
			cornersym,
			into_edge8_perm,
			rawtosym,
		}
	}

	fn heuristic_at(&self, c: &CoordPhase1) -> u8 {
		let (dz, sym) = self
			.rawtosym
			.get(c.edge_ori + (c.udslice_sorted / 24) * EDGE_ORI)
			.expect("Could not get heuristics!");
		let csym = self.cornersym[c.corner_ori][*sym as usize] as usize;
		let coord = csym + CORNER_ORI * *dz as usize;

		match self.heuristic.get(coord) {
			Some(t) => t,
			None => panic!("Index ist {} ({},{})", coord, csym, dz),
		}
	}

	pub fn is_solved(&self, c: &CoordPhase1) -> bool {
		c.corner_ori == 0 && c.edge_ori == 0 && c.udslice_sorted < 24
	}

	pub fn apply_turn(&self, c: &mut CoordPhase1, i: usize) {
		c.corner_ori = self.corner_ori_move[c.corner_ori][i] as usize;
		c.edge_ori = self.edge_ori_move[c.edge_ori][i] as usize;
		c.corner_perm = self.corner_perm_move[c.corner_perm][i] as usize;

		c.udslice_sorted = self.slice_move[c.udslice_sorted][i] as usize;
		c.fbslice_sorted = self.slice_move[c.fbslice_sorted][i] as usize;
		c.rlslice_sorted = self.slice_move[c.rlslice_sorted][i] as usize;
	}

	pub fn succ(&self, coord: &CoordPhase1) -> Vec<(u8, usize)> {
		let cur = self.heuristic_at(coord);

		let mut out: Vec<(_, _)> = (0..self.turns.len())
			.into_par_iter()
			.map(|i| {
				let mut c = coord.clone();
				self.apply_turn(&mut c, i);
				let h = self.heuristic_at(&c);

				// How many turns you would waste when choosing this path
				// if (h+1) % 3 == cur, you waste no turn. (cur+1) % 3 == h wastes 2 turns.
				let w = (4 - cur + h) % 3;
				(w, i)
			})
			.filter(|(w, _)| *w == 0) // this creates shorter solutions
			.collect();
		out.sort();

		out
	}

	/// Convert phase 1 coordinate into phase 2 coordinate
	pub fn convert_to_phase2_coord(&self, c: &CoordPhase1) -> CoordPhase2 {
		// Use fbslice and rlslice to get egde8_perm
		let edge8 = self.into_edge8_perm[c.fbslice_sorted][c.rlslice_sorted % 24] as usize;

		CoordPhase2 {
			corner_perm: c.corner_perm,
			edge8_perm: edge8,
			udslice_sorted: c.udslice_sorted,
		}
	}
}

#[derive(Clone, Debug)]
struct CoordPhase2 {
	corner_perm: usize,
	edge8_perm: usize,
	udslice_sorted: usize,
}

#[derive(Serialize, Deserialize)]
struct DataPhase2 {
	turns: Vec<Turn>,

	// Movetables
	corner_perm_move: Movetable,
	edge8_perm_move: Movetable,
	slice_move: Movetable,

	// Heuristics
	heuristic: Heuristics,

	// Symtable
	edgesym: Symtable,

	// Rawtosym
	rawtosym: RawToSymTable,
}

impl DataPhase2 {
	/// Get all the tables needed for phase 2
	pub fn new(turns: Vec<Turn>) -> Self {
		let corner_perm_move = create_movetable(
			CORNER_PERM,
			&turns,
			get_corner_perm_coord,
			cube_from_corner_perm,
		);

		let edge8_perm_move =
			create_movetable(EDGE8_PERM, &turns, get_edge8_perm, cube_from_edge_perm);
		let slice_move = create_movetable(
			24, // = 4!
			&turns,
			get_udslice_sorted_coord,
			cube_from_udslice_sorted,
		);

		let corner_perm_toraw = create_symtoraw(
			CORNER_PERM,
			SYM_CORNER_PERM,
			get_corner_perm_coord,
			cube_from_corner_perm,
		);

		let corner_perm_symstate = gen_symstate(
			&corner_perm_toraw,
			get_corner_perm_coord,
			cube_from_corner_perm,
		);

		let stable = create_sym_movetable(
			&corner_perm_toraw,
			&turns,
			get_corner_perm_coord,
			cube_from_corner_perm,
		);

		let edge8_symtable = create_symtable(EDGE8_PERM, 16, get_edge8_perm, cube_from_edge_perm);

		let heuristic = gen_heuristics(
			&edge8_perm_move,
			&stable,
			&edge8_symtable,
			&corner_perm_symstate,
		);

		let corner_perm_rawtosym =
			create_rawtosym(CORNER_PERM, get_corner_perm_coord, cube_from_corner_perm);

		Self {
			turns,
			corner_perm_move,
			edge8_perm_move,
			slice_move,
			heuristic,
			edgesym: edge8_symtable,
			rawtosym: corner_perm_rawtosym,
		}
	}

	fn heuristic_at(&self, c: &CoordPhase2) -> u8 {
		let (dz, sym) = self.rawtosym[c.corner_perm];
		let esym = self.edgesym[c.edge8_perm][sym as usize] as usize;
		let coord = esym + EDGE8_PERM * dz as usize;

		self.heuristic.get(coord).unwrap()
	}

	pub fn is_solved(&self, c: &CoordPhase2) -> bool {
		c.corner_perm == 0 && c.edge8_perm == 0 && c.udslice_sorted == 0
	}

	/// Apply a turn to the given coordinate
	pub fn apply_turn(&self, c: &mut CoordPhase2, i: usize) {
		c.corner_perm = self.corner_perm_move[c.corner_perm][i] as usize;
		c.edge8_perm = self.edge8_perm_move[c.edge8_perm][i] as usize;
		c.udslice_sorted = self.slice_move[c.udslice_sorted][i] as usize;
	}

	pub fn succ(&self, coord: &CoordPhase2) -> Vec<(u8, usize)> {
		let cur = self.heuristic_at(coord);

		let mut out: Vec<(_, _)> = (0..self.turns.len())
			.into_par_iter()
			.map(|i| {
				let mut c = coord.clone();
				self.apply_turn(&mut c, i);
				let h = self.heuristic_at(&c);

				// Same as in DataPhase2::succ
				let w = (4 - cur + h) % 3;
				(w, i)
			})
			.collect();
		out.sort();

		out
	}
}

#[derive(Serialize, Deserialize)]
/// Struct containing all data tables used for the Kociemba algorithm
pub struct KociembaData {
	phase1: DataPhase1,
	phase2: DataPhase2,
}

impl KociembaData {
	pub fn generate(advanced_turns: bool) -> Self {
		let turns_phase1 = if advanced_turns {
			crate::cube::turn::all_turns()
		} else {
			parse_turns("U U2 U' D D2 D' B B2 B' F F2 F' L L2 L' R R2 R'").unwrap()
		};

		let turns_phase2 = if advanced_turns {
			parse_turns("U U2 U' D D2 D' B2 F2 L2 R2 M2 E E' E2 S2 EC EC'").unwrap()
		} else {
			parse_turns("U U2 U' D D2 D' B2 F2 L2 R2").unwrap()
		};

		Self {
			phase1: DataPhase1::new(turns_phase1),
			phase2: DataPhase2::new(turns_phase2),
		}
	}

	pub fn load(path: impl Into<String>) -> std::io::Result<Self> {
		let mut file = std::fs::File::open(path.into())?;
		let mut buf = vec![];
		file.read_to_end(&mut buf)?;
		let decoded: Self = bincode::deserialize(&buf).map_err(|e| {
			std::io::Error::new(
				std::io::ErrorKind::InvalidData,
				format!("Could not deserialize: {}", e),
			)
		})?;
		Ok(decoded)
	}

	pub fn save(&self, path: impl Into<String>) -> std::io::Result<()> {
		let mut file = std::fs::File::create(path.into())?;
		let encode: Vec<u8> = bincode::serialize(&self).map_err(|e| {
			std::io::Error::new(
				std::io::ErrorKind::InvalidInput,
				format!("Could not serialize: {}", e),
			)
		})?;
		file.write_all(&encode)?;

		Ok(())
	}
}

/// Solver struct which uses Kociemba two phase algorithm to solve the
/// cube.
pub struct Solver {
	data: KociembaData,

	sol: Option<Vec<Turn>>,
	start: std::time::Instant,
}

/// Maximum solving time for the solving algorithm.
const MAX_SOLVING_TIME: u128 = 100;

impl Solver {
	/// Initialise the solver
	pub fn new(data: KociembaData) -> Self {
		Self {
			data,
			sol: None,
			start: std::time::Instant::now(),
		}
	}

	/// Solve the given cube and return a sequence if possible
	pub fn solve<T>(&mut self, cube: T) -> Option<Vec<Turn>>
	where
		T: TryInto<CubieCube>,
	{
		let cube = match cube.try_into() {
			Ok(cube) => cube,
			Err(_) => return None,
		};

		if !cube.is_solvable() {
			return None;
		}

		self.start = std::time::Instant::now();

		let coord = CoordPhase1::new(&cube);
		let mut path = vec![];

		for d in 0..=12 {
			if self.search_phase1(coord.clone(), &mut path, d) {
				break;
			}
		}
		let res = self.sol.clone();
		self.sol = None;
		res
	}

	/// Phase 1 search function
	fn search_phase1(&mut self, c: CoordPhase1, path: &mut Vec<Turn>, depth: isize) -> bool {
		if let Some(v) = &self.sol {
			if MAX_SOLVING_TIME < self.start.elapsed().as_millis() {
				return true;
			}

			if v.len() <= path.len() {
				return false;
			}
		}

		// Phase 1 done, search a solution in phase 2
		if self.data.phase1.is_solved(&c) {
			let next = self.data.phase1.convert_to_phase2_coord(&c);

			for d in 0..=18 {
				if self.search_phase2(next.clone(), path, d) {
					break;
				}
			}
		}

		if depth < 0 {
			return false;
		}

		// Sort all neighbouring cubes by how many turns you'll waste
		for (w, i) in self.data.phase1.succ(&c) {
			path.push(self.data.phase1.turns[i]);

			let mut next = c.clone();
			self.data.phase1.apply_turn(&mut next, i);

			if self.search_phase1(next, path, depth - w as isize) {
				return true;
			}

			path.pop();
		}

		false
	}

	/// Phase 2 search function
	fn search_phase2(&mut self, c: CoordPhase2, path: &mut Vec<Turn>, depth: isize) -> bool {
		if let Some(v) = &self.sol {
			if MAX_SOLVING_TIME < self.start.elapsed().as_millis() {
				return true;
			}

			if v.len() <= path.len() {
				return true;
			}
		}

		// Solved the cube, terminate phase 2 and search for other solutions
		if self.data.phase2.is_solved(&c) {
			self.sol = Some(path.clone());
			return true;
		}

		if depth < 0 {
			return false;
		}

		for (w, i) in self.data.phase2.succ(&c) {
			path.push(self.data.phase2.turns[i]);

			let mut next = c.clone();
			self.data.phase2.apply_turn(&mut next, i);

			let res = self.search_phase2(next, path, depth - w as isize);
			path.pop();

			if res {
				return true;
			}
		}

		false
	}
}

// ===== Coordinate get functions =====
// These functions extract the specified coordinate from
// the given cube.
// See more: https://kociemba.org/math/coordlevel.htm

fn get_corner_ori_coord(cube: &CubieCube) -> usize {
	cube.get_corner_orientation_coord()
}

fn get_edge_ori_coord(cube: &CubieCube) -> usize {
	cube.get_edge_orientation_coord()
}

fn get_corner_perm_coord(cube: &CubieCube) -> usize {
	cube.get_corner_perm_coord()
}

fn get_edge8_perm(cube: &CubieCube) -> usize {
	let perm: Vec<_> = cube
		.edges
		.iter()
		.take(8)
		.map(|(e, _)| *e as usize)
		.collect();

	map_permutation(&perm)
}

fn get_flip_udslice_coord(cube: &CubieCube) -> usize {
	let edge = cube.get_edge_orientation_coord();
	let chosen: Vec<_> = Edge::iter()
		.map(|pos| {
			let (e, _) = cube.edge_at(pos);
			UDSLICE_EDGES.contains(&e)
		})
		.collect();

	let udslice_coord = map_nck(&chosen);
	udslice_coord * EDGE_ORI + edge
}

fn get_slice_coord_sorted(cube: &CubieCube, slice: &[Edge]) -> usize {
	let vec: Vec<_> = cube.edges.iter().map(|(e, _)| slice.contains(e)).collect();
	let ord: Vec<_> = cube
		.edges
		.iter()
		.filter(|(e, _)| slice.contains(e))
		.map(|(e, _)| EDGE_SLICE_INDEX[*e as usize])
		.collect();
	map_nck(&vec) * 24 + map_permutation(&ord)
}

fn get_udslice_sorted_coord(cube: &CubieCube) -> usize {
	get_slice_coord_sorted(cube, &UDSLICE_EDGES)
}

fn get_rlslice_sorted_coord(cube: &CubieCube) -> usize {
	get_slice_coord_sorted(cube, &RLSLICE_EDGES)
}

fn get_fbslice_sorted_coord(cube: &CubieCube) -> usize {
	get_slice_coord_sorted(cube, &FBSLICE_EDGES)
}

// ===== Cube form coordinate functions =====
// The inverse function of those above.
// Instead of extracting a coordinate, it produces a cube
// containing the coordinate

fn cube_from_corner_ori(coord: usize) -> CubieCube {
	let mut cube = CubieCube::new();
	cube.set_corner_orientation(coord);
	cube
}

fn cube_from_edge_ori(coord: usize) -> CubieCube {
	let mut cube = CubieCube::new();
	cube.set_edge_orientation(coord);
	cube
}

/// Return a cube distributing the edges according to coord
/// The idx is the idx-th type of n choose k
fn cube_from_edge_pos(edges: Vec<Edge>, coord: usize) -> CubieCube {
	let mut cube = CubieCube::new();

	let chosen = get_nck(NUM_EDGES, edges.len(), coord);

	let mut m = 0;
	let mut n = 0;
	let nonmiddle: Vec<_> = Edge::iter().filter(|e| !edges.contains(e)).collect();

	for (i, chosen) in chosen.into_iter().enumerate() {
		if chosen {
			cube.edges[i].0 = edges[m];
			m += 1;
		} else {
			cube.edges[i].0 = nonmiddle[n];
			n += 1;
		}
	}

	cube
}

/// Get the cube with:
/// - udslice coord = idx / EDGE_ORI
/// - edge ori coord = idx % EDGE_ORI
fn cube_from_flip_udslice(idx: usize) -> CubieCube {
	let edge_ori_coord = idx % EDGE_ORI;
	let udslice_coord = idx / EDGE_ORI;

	let mut cube = cube_from_edge_ori(edge_ori_coord);
	let edge_cube = cube_from_edge_pos(UDSLICE_EDGES.to_vec(), udslice_coord);
	for i in 0..NUM_EDGES {
		cube.edges[i].0 = edge_cube.edges[i].0;
	}

	cube
}

fn cube_from_udslice_sorted(coord: usize) -> CubieCube {
	let pos = coord / 24;
	let ord = coord % 24;

	let slice = permute_vec(UDSLICE_EDGES.to_vec(), ord);
	cube_from_edge_pos(slice, pos)
}

fn cube_from_fbslice_sorted(coord: usize) -> CubieCube {
	let pos = coord / 24;
	let ord = coord % 24;

	let slice = permute_vec(FBSLICE_EDGES.to_vec(), ord);
	cube_from_edge_pos(slice, pos)
}

/// Cube from corner permutation coord
fn cube_from_corner_perm(coord: usize) -> CubieCube {
	let mut cube = CubieCube::new();
	cube.set_corner_permutation(coord);
	cube
}

/// Cube from edge permutation coord
fn cube_from_edge_perm(coord: usize) -> CubieCube {
	let mut cube = CubieCube::new();
	cube.set_edge_permutation(coord);
	cube
}

// ===== Table Generating =====

fn create_movetable(
	num_states: usize,
	moves: &Vec<Turn>,
	to_coord: fn(&CubieCube) -> usize,
	from_coord: fn(usize) -> CubieCube,
) -> Movetable {
	// For each possible coordinate, calculate their neighbours
	(0..num_states)
		.into_par_iter()
		.map(|coord| {
			// Create cube from current coordinate
			let cube = from_coord(coord);

			// Apply each turn to the cube and save the corresponding index
			moves
				.par_iter()
				.map(|turn| {
					let mut ncube = cube.clone();
					ncube.apply_turn(*turn);
					to_coord(&ncube) as u16
				})
				.collect()
		})
		.collect()
}

/// Creates a list where v[i] is an element of symmetry group i
fn create_symtoraw(
	num_states: usize,
	num_symmetry_states: usize,
	to_coord: fn(&CubieCube) -> usize,
	from_coord: fn(usize) -> CubieCube,
) -> SymToRawTable {
	let mut out = vec![0u32; num_symmetry_states];
	let mut used = bit_set::BitSet::with_capacity(num_states);
	let mut symidx = 0;

	for idx in 0..num_states {
		if used.contains(idx) {
			continue;
		}

		// Generate cube from index
		let cube: CubieCube = from_coord(idx);

		out[symidx] = idx as u32;
		symidx += 1;

		used.insert(idx);
		for sym in 1..16 {
			// Generate new cube
			let csym = cube.get_symmetry(sym);
			let n = to_coord(&csym); // get coordinate

			// Say that this cube is already used
			used.insert(n);
		}
	}

	out
}

/// Creates a list where v[i] = symmetry group of i
fn create_rawtosym(
	num_states: usize,
	to_coord: fn(&CubieCube) -> usize,
	from_coord: fn(usize) -> CubieCube,
) -> RawToSymTable {
	const UNVISITED: u16 = u16::MAX;
	let mut out = vec![(UNVISITED, 0); num_states];
	let mut symidx = 0;

	for idx in 0..num_states {
		if out[idx].0 != UNVISITED {
			continue;
		}

		// Generate cube from index
		let cube: CubieCube = from_coord(idx);

		out[idx] = (symidx as u16, 0);
		for sym in 1..16 {
			// Generate new cube and get coord
			let csym = cube.get_symmetry(sym);
			let n = to_coord(&csym);

			out[n] = (symidx as u16, sym as u8);
		}
		symidx += 1;
	}

	out
}

fn create_symtable(
	num_states: usize,
	num_symmetries: usize,
	to_coord: fn(&CubieCube) -> usize,
	from_coord: fn(usize) -> CubieCube,
) -> Symtable {
	(0..num_states)
		.into_par_iter()
		.map(|idx| {
			let cube = from_coord(idx);

			(0..num_symmetries)
				.into_par_iter()
				.map(|symidx| {
					let csym = cube.get_inv_symmetry(symidx);
					to_coord(&csym) as u16
				})
				.collect()
		})
		.collect()
}

fn create_sym_movetable(
	symtoraw: &SymToRawTable,
	moves: &Vec<Turn>,
	to_coord: fn(&CubieCube) -> usize,
	from_coord: fn(usize) -> CubieCube,
) -> SymMovetable {
	// Number of entries
	let n = symtoraw.len();

	(0..n)
		.into_par_iter()
		.map(|idx| {
			// Create cube from current symtoraw[idx]
			let cube = from_coord(symtoraw[idx] as usize);

			moves
				.par_iter()
				.map(|turn| {
					// Apply the move and get it's sym coordinate
					let mut ncube = cube.clone();
					ncube.apply_turn(*turn);

					let (dst, sym) = get_sym_class(&ncube, symtoraw, to_coord).unwrap();
					(dst as u16, sym as u8)
				})
				.collect()
		})
		.collect()
}

fn get_sym_class(
	cube: &CubieCube,
	symtoraw: &SymToRawTable,
	to_coord: fn(&CubieCube) -> usize,
) -> Option<(usize, usize)> {
	for sym in 0..NUM_SYMMETRIES {
		let c = cube.get_inv_symmetry(sym);
		let dst = to_coord(&c) as u32;

		if let Ok(k) = symtoraw.binary_search(&dst) {
			return Some((k, sym));
		}
	}
	None
}

fn gen_symstate(
	symtoraw: &SymToRawTable,
	to_coord: fn(&CubieCube) -> usize,
	from_coord: fn(usize) -> CubieCube,
) -> SymState {
	(0..symtoraw.len())
		.into_par_iter()
		.map(|i| {
			let raw = symtoraw[i] as usize;
			let cube = from_coord(raw);

			// if a * sym[i] = a * sym[j], we have more
			// possible moves todo afterward!
			let mut out = 0;
			for sym in 1..16 {
				let csym = cube.get_symmetry(sym);
				let idx = to_coord(&csym);

				if idx == raw {
					out |= 1 << sym;
				}
			}
			out as u32
		})
		.collect()
}

/// Generate heurisitcs
///
/// Both phase 1 and 2 use a raw coordinate in combination
/// with a sym coordinate.
fn gen_heuristics(
	movetable: &Movetable,
	symmovetable: &SymMovetable,
	symtable: &Symtable,
	symstate: &SymState,
) -> Heuristics {
	// The length of the output table
	let n = symmovetable.len() * movetable.len();

	// The number of non-sym coords
	let num_raw_coords = movetable.len();
	let numturns = movetable[0].len();

	const UNVISITED: u8 = u8::MAX;
	let mut out = vec![UNVISITED; n];
	out[0] = 0;
	let mut cnt = 1;

	for m in 0..100 {
		if cnt == n {
			break;
		}

		let bef = cnt;

		#[cfg(debug_assertions)]
		let now = std::time::Instant::now();

		#[cfg(debug_assertions)]
		{
			let percent = cnt as f32 / (n as f32) * 100f32;
			println!("Generating depth: {} ({}%)", m, percent);
		}

		// If more entries are unvisited than visited
		let forward_search = cnt < n / 2;

		if forward_search {
			// Forward search
			// From all visited coords (with depth = m) find unvisited coords
			let bound = if m == 0 { 1 } else { n };
			for idx in 0..bound {
				if out[idx] != m {
					continue;
				}

				// Extract coordinates from fused coordinate
				let c0 = idx % num_raw_coords;
				let csym0 = idx / num_raw_coords;

				for i in 0..numturns {
					let (sym_coord, sym) = symmovetable[csym0][i];
					let c1 = movetable[c0][i] as usize;
					let coord = symtable[c1][sym as usize]; // Apply symmetry to non-sym coordinate

					// Calculate the coordinate when applying this turn
					let dst = coord as usize + sym_coord as usize * num_raw_coords;

					if out[dst] != UNVISITED {
						continue;
					}
					out[dst] = m + 1;
					cnt += 1;

					let state = symstate[sym_coord as usize];
					if state <= 1 {
						continue;
					}

					for j in 1..16 {
						if (state >> j) & 1 == 1 {
							let ddx = symtable[coord as usize][j as usize];
							let dst = ddx as usize + sym_coord as usize * num_raw_coords;

							if out[dst] != UNVISITED {
								continue;
							}
							out[dst] = m + 1;
							cnt += 1;
						}
					}
				}
			}
		} else {
			// Backward search
			// From all unvisited coords, find a visited coord with depth = m
			for idx in 0..n {
				if out[idx] != UNVISITED {
					continue;
				}

				let c0 = idx % num_raw_coords;
				let sym0 = idx / num_raw_coords;

				for i in 0..numturns {
					let (sym_coord, sym) = symmovetable[sym0][i];
					let c1 = movetable[c0][i] as usize;
					let coord = symtable[c1][sym as usize]; // Apply symmetry to non-sym coordinate

					// Calculate the coordinate when applying this turn
					let dst = coord as usize + sym_coord as usize * num_raw_coords;

					if out[dst] != m {
						continue;
					}
					out[idx] = m + 1;
					cnt += 1;
					break;
				}
			}
		}

		#[cfg(debug_assertions)]
		println!("Time elapsed: {:.?}", now.elapsed());

		// If nothing has changed, stop generating
		if bef == cnt {
			break;
		}
	}

	Heuristics::from(out)
}

// ===== Slices into edge8_perm table =====
// This table is used to create the phase 2 coordinate
// from the helper coordinates in phase 1

fn get_into_edge8perm() -> Vec<Vec<u16>> {
	(0..SLICE_SORTED)
		.into_par_iter()
		.map(|idx| {
			let mut cube = cube_from_fbslice_sorted(idx);

			// Check if the udslice only contains edges from the udslice
			if UDSLICE_EDGES
				.iter()
				.any(|e| !UDSLICE_EDGES.contains(&cube.edges[*e as usize].0))
			{
				return vec![u16::MAX; 24];
			}

			// Get the indices where the RL-slice edges are
			let indices: Vec<_> = cube
				.edges
				.iter()
				.take(8)
				.enumerate()
				.filter(|(_, (e, _))| RLSLICE_EDGES.contains(e))
				.map(|(i, _)| i)
				.collect();

			// For each permutation, apply a rlslice order
			(0..24)
				.map(|j| {
					let rlslice_edges = permute_vec(RLSLICE_EDGES.to_vec(), j);

					// Permute the RLSLICE indices
					for (i, ele) in indices.iter().enumerate() {
						cube.edges[*ele].0 = rlslice_edges[i];
					}

					cube.get_edge_permutation_coord() as u16
				})
				.collect()
		})
		.collect()
}
