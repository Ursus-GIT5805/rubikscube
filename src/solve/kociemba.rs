use std::io::{Read, Result, Write};

use lazy_static::lazy_static;
use rayon::prelude::*;
use strum::IntoEnumIterator;

use crate::{arraycube::ArrayCube, cubiecube::*, parse_turns, Edge, RubiksCube, Turn, NUM_EDGES};

use crate::math::*;

/// v[coord][i] is the coordinate when applying move i on coord
type Movetable = Vec<Vec<u16>>;

/// v[coord][i] is the coordinate of the i-th symmetry of coord
type Symtable = Vec<Vec<u16>>;

/// v[symcoord][i] = (dst, sym) results by applying the move i on the symcoordinate "symcoord"
/// where dst: Is the resulting sym-coordinate
/// where sym: Is the sym-th symmetry of the to raw coordinate of symcoord
type SymMovetable = Vec<Vec<(u16, u8)>>;

/// A mapper vector where v[i] is the raw coordinate of the
/// representant cube of symmetry class i
type SymToRawTable = Vec<u32>;

type RawToSymTable = Vec<(u16, u8)>;

type SymState = Vec<u32>;

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

fn get_turns_phase1(advanced_turns: bool) -> Vec<Turn> {
	if advanced_turns {
		return crate::cube::turn::all_turns();
	}
	parse_turns("U U2 U' D D2 D' B B2 B' F F2 F' L L2 L' R R2 R'").unwrap()
}

fn get_turns_phase2(advanced_turns: bool) -> Vec<Turn> {
	if advanced_turns {
		return parse_turns("U U2 U' D D2 D' B2 F2 L2 R2 M2 E E' E2 S2 EC EC'").unwrap();
	}
	parse_turns("U U2 U' D D2 D' B2 F2 L2 R2").unwrap()
}

lazy_static! {
	static ref toraw: SymToRawTable = gen_symmetrytoraw();
	static ref toraw2: SymToRawTable = gen2_symmetrytoraw();
	static ref rlslice: Vec<Edge> = vec![Edge::UF, Edge::DF, Edge::DB, Edge::UB];
	static ref fbslice: Vec<Edge> = vec![Edge::UR, Edge::DR, Edge::DL, Edge::UL];
	static ref udslice: Vec<Edge> = vec![Edge::FR, Edge::BR, Edge::BL, Edge::FL];
}

// Helper list to map the index of the edge in an edge-slice
const EDGE_SLICE_INDEX: [usize; NUM_EDGES] = [0, 0, 3, 3, 1, 1, 2, 2, 0, 1, 2, 3];

const SYM_LEN: usize = 64430;
const UDSLICE: usize = 495;

const EDGE8_PERM: usize = 40320;
const SYM2_LEN: usize = 2768;

const SLICE_SORTED: usize = 495 * 24;

fn get_edge8_perm(cube: &CubieCube) -> usize {
	cube.get_edge8_permutation_coord()
}

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
	pub fn new(turns: Vec<Turn>) -> Self {
		let corner_ori_move = create_movetable(
			CORNER_ORI,
			turns.clone(),
			get_corner_ori_coord,
			cube_from_corner_ori,
		);

		let corner_perm_move = create_movetable(
			CORNER_PERM,
			turns.clone(),
			get_corner_perm_coord,
			cube_from_corner_perm,
		);

		let edge_ori_move = create_movetable(
			EDGE_ORI,
			turns.clone(),
			get_edge_ori_coord,
			cube_from_edge_ori,
		);

		let slice_move = create_movetable(
			24 * 495,
			turns.clone(),
			get_udslice_sorted_coord,
			cube_from_udslice_sorted,
		);

		let heuristic = gen_phase1_heuristics(false);

		let cornersym = gen_corner_ori_symtable();

		let rawtosym = create_rawtosym_list(
			EDGE_ORI * 495,
			get_flipudslice_coord,
			cube_from_udslice_edge_idx,
		);

		let into_edge8_perm = gen_toedge8perm_table();

		Self {
			turns,
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
		let (dz, sym) = self.rawtosym[c.edge_ori + (c.udslice_sorted / 24) * EDGE_ORI];
		let csym = self.cornersym[c.corner_ori][sym as usize] as usize;
		let coord = csym + CORNER_ORI * dz as usize;

		match self.heuristic.get(coord) {
			Some(t) => t,
			None => {
				panic!("Index ist {} ({},{})", coord, csym, dz)
			}
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

				let w = if (h + 1) % 3 == cur {
					0
				} else if h == cur {
					1
				} else {
					2
				};

				(w, i)
			})
			.filter(|(w, _)| *w == 0) // this creates shorter solutions...
			.collect();
		out.sort();

		out
	}

	pub fn convert_to_phase2_coord(&self, c: &CoordPhase1) -> CoordPhase2 {
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
	pub fn new(turns: Vec<Turn>) -> Self {
		let corner_perm_move = create_movetable(
			CORNER_PERM,
			turns.clone(),
			get_corner_perm_coord,
			cube_from_corner_perm,
		);

		let edge8_perm_move = create_movetable(
			EDGE8_PERM,
			turns.clone(),
			get_edge8_perm,
			cube_from_edge_perm,
		);

		let slice_move = create_movetable(
			24,
			turns.clone(),
			get_udslice_sorted_coord,
			cube_from_udslice_sorted,
		);

		let heuristic = gen_phase2_heuristics(false);

		let edgesymm = gen2_edge_perm_symtable();

		let rawtosym =
			create_rawtosym_list(CORNER_PERM, get_corner_perm_coord, cube_from_corner_perm);

		Self {
			turns,
			corner_perm_move,
			edge8_perm_move,
			slice_move,
			heuristic,
			edgesym: edgesymm,
			rawtosym,
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
				c.corner_perm = self.corner_perm_move[c.corner_perm][i] as usize;
				c.edge8_perm = self.edge8_perm_move[c.edge8_perm][i] as usize;

				let h = self.heuristic_at(&c);

				let w = if (h + 1) % 3 == cur {
					0
				} else if h == cur {
					1
				} else {
					2
				};

				(w, i)
			})
			.collect();
		out.sort();

		out
	}
}

struct Solver {
	phase1: DataPhase1,
	phase2: DataPhase2,

	sol: Option<Vec<Turn>>,
	start: std::time::Instant,
}

const MAX_SOLVING_TIME: u128 = 100;

impl Solver {
	pub fn new(advanced_turns: bool) -> Self {
		let turns_phase1 = get_turns_phase1(advanced_turns);
		let turns_phase2 = get_turns_phase2(advanced_turns);

		Self {
			phase1: DataPhase1::new(turns_phase1),
			phase2: DataPhase2::new(turns_phase2),

			sol: None,
			start: std::time::Instant::now(),
		}
	}

	pub fn solve(&mut self, cube: CubieCube) -> Option<Vec<Turn>> {
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

	fn search_phase1(&mut self, c: CoordPhase1, path: &mut Vec<Turn>, depth: isize) -> bool {
		if let Some(v) = &self.sol {
			if MAX_SOLVING_TIME < self.start.elapsed().as_millis() {
				return true;
			}

			if v.len() <= path.len() {
				return false;
			}
		}

		if self.phase1.is_solved(&c) {
			let next = self.phase1.convert_to_phase2_coord(&c);

			for d in 0..=18 {
				if self.search_phase2(next.clone(), path, d) {
					break;
				}
			}
		}

		if depth < 0 {
			return false;
		}

		for (w, i) in self.phase1.succ(&c) {
			path.push(self.phase1.turns[i]);

			let mut next = c.clone();
			self.phase1.apply_turn(&mut next, i);

			if self.search_phase1(next, path, depth - w as isize) {
				return true;
			}

			path.pop();
		}

		false
	}

	fn search_phase2(&mut self, c: CoordPhase2, path: &mut Vec<Turn>, depth: isize) -> bool {
		if let Some(v) = &self.sol {
			if MAX_SOLVING_TIME < self.start.elapsed().as_millis() {
				return true;
			}

			if v.len() <= path.len() {
				return true;
			}
		}

		if self.phase2.is_solved(&c) {
			self.sol = Some(path.clone());
			return true;
		}

		if depth < 0 {
			return false;
		}

		for (w, i) in self.phase2.succ(&c) {
			path.push(self.phase2.turns[i]);

			let mut next = c.clone();
			self.phase2.apply_turn(&mut next, i);

			let res = self.search_phase2(next, path, depth - w as isize);
			path.pop();

			if res {
				return true;
			}
		}

		false
	}
}

fn get_flipudslice_coord(cube: &CubieCube) -> usize {
	let edge = cube.get_edge_orientation_coord();
	let udslice_coord = cube.get_udslice_coord();
	udslice_coord * EDGE_ORI + edge
}

fn get_corner_ori_coord(cube: &CubieCube) -> usize {
	cube.get_corner_orientation_coord()
}

fn get_edge_ori_coord(cube: &CubieCube) -> usize {
	cube.get_edge_orientation_coord()
}

fn get_corner_perm_coord(cube: &CubieCube) -> usize {
	cube.get_corner_perm_coord()
}

fn cube_from_edge_ori(coord: usize) -> CubieCube {
	let mut cube = CubieCube::new();
	cube.set_edge_orientation(coord);
	cube
}

fn cube_from_corner_ori(coord: usize) -> CubieCube {
	let mut cube = CubieCube::new();
	cube.set_corner_orientation(coord);
	cube
}

fn cube_from_edge_pos_idx(edges: Vec<Edge>, idx: usize) -> CubieCube {
	let mut cube = CubieCube::new();

	let chosen = get_nck(NUM_EDGES, edges.len(), idx);

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
fn cube_from_udslice_edge_idx(idx: usize) -> CubieCube {
	let edge_ori_coord = idx % EDGE_ORI;
	let udslice_coord = idx / EDGE_ORI;

	let mut cube = cube_from_edge_ori(edge_ori_coord);
	let edge_cube = cube_from_edge_pos_idx(udslice.to_vec(), udslice_coord);
	for i in 0..NUM_EDGES {
		cube.edges[i].0 = edge_cube.edges[i].0;
	}

	cube
}

fn cube_from_udslice_sorted(idx: usize) -> CubieCube {
	let pos = idx / 24;
	let ord = idx % 24;

	let slice = permute_vec(udslice.to_vec(), ord);
	cube_from_edge_pos_idx(slice, pos)
}

fn cube_from_fbslice_idx(idx: usize) -> CubieCube {
	let pos = idx / 24;
	let ord = idx % 24;

	let slice = permute_vec(fbslice.to_vec(), ord);
	cube_from_edge_pos_idx(slice, pos)
}

fn cube_from_corner_perm(coord: usize) -> CubieCube {
	let mut cube = CubieCube::new();
	cube.set_corner_permutation(coord);
	cube
}

fn cube_from_edge_perm(coord: usize) -> CubieCube {
	let mut cube = CubieCube::new();
	cube.set_edge_permutation(coord);
	cube
}

// ===== Table Generating =====

/// Create a movetable
fn create_movetable(
	num_states: usize,
	moves: Vec<Turn>,
	to_idx: fn(&CubieCube) -> usize,
	from_idx: fn(usize) -> CubieCube,
) -> Movetable {
	(0..num_states)
		.into_par_iter()
		.map(|idx| {
			// Create cube from current idx
			let cube = from_idx(idx);

			// Apply each turn to the cube and save the corresponding index
			moves
				.par_iter()
				.map(|turn| {
					let mut ncube = cube.clone();
					ncube.apply_turn(*turn);
					to_idx(&ncube) as u16
				})
				.collect()
		})
		.collect()
}

fn create_symmetrytoraw_list(
	num_states: usize,
	num_symmetry_states: usize,
	to_idx: fn(&CubieCube) -> usize,
	from_idx: fn(usize) -> CubieCube,
) -> SymToRawTable {
	let mut out = vec![0u32; num_symmetry_states];
	let mut used = bit_set::BitSet::with_capacity(num_states);
	let mut symidx = 0;

	for idx in 0..num_states {
		if used.contains(idx) {
			continue;
		}

		// Generate cube from index
		let cube: CubieCube = from_idx(idx);

		out[symidx] = idx as u32;
		symidx += 1;

		used.insert(idx);
		for sym in 1..16 {
			// Generate new cube
			let csym = get_symmetry(&cube, sym);

			// Parse coordinate of cube
			let n = to_idx(&csym);

			// Say that this cube is already used
			used.insert(n);
		}
	}

	out
}

fn create_rawtosym_list(
	num_states: usize,
	to_idx: fn(&CubieCube) -> usize,
	from_idx: fn(usize) -> CubieCube,
) -> RawToSymTable {
	const UNVISITED: u16 = u16::MAX;
	let mut out = vec![(u16::MAX, u8::MAX); num_states];
	let mut symidx = 0;

	for idx in 0..num_states {
		if out[idx].0 != UNVISITED {
			continue;
		}

		// Generate cube from index
		let cube: CubieCube = from_idx(idx);

		out[idx] = (symidx as u16, 0);
		for sym in 1..16 {
			// Generate new cube and get coord
			let csym = get_symmetry(&cube, sym);
			let n = to_idx(&csym);

			out[n] = (symidx as u16, sym as u8);
		}
		symidx += 1;
	}

	out
}

fn create_symtable(
	num_states: usize,
	num_symmetries: usize,
	to_idx: fn(&CubieCube) -> usize,
	from_idx: fn(usize) -> CubieCube,
) -> Symtable {
	(0..num_states)
		.into_par_iter()
		.map(|idx| {
			let cube = from_idx(idx);

			(0..num_symmetries)
				.into_par_iter()
				.map(|symidx| {
					let csym = get_symmetry_inv(&cube, symidx);
					to_idx(&csym) as u16
				})
				.collect()
		})
		.collect()
}

fn create_sym_movetable(
	symtoraw: &SymToRawTable,
	from_idx: fn(usize) -> CubieCube,
	to_idx: fn(&CubieCube) -> usize,
	moves: Vec<Turn>,
) -> SymMovetable {
	// Number of entries
	let n = symtoraw.len();

	(0..n)
		.into_par_iter()
		.map(|idx| {
			// Create cube from current symtoraw[idx]
			let cube = from_idx(symtoraw[idx] as usize);

			moves
				.par_iter()
				.map(|turn| {
					let ncube = {
						let mut ncube = cube.clone();
						ncube.apply_turn(*turn);
						ncube
					};

					let (dst, sym) = get_sym_class(&ncube, symtoraw, to_idx).unwrap();
					(dst as u16, sym as u8)
				})
				.collect()
		})
		.collect()
}

fn get_sym_class(
	cube: &CubieCube,
	symtoraw: &SymToRawTable,
	to_idx: fn(&CubieCube) -> usize,
) -> Option<(usize, usize)> {
	for sym in 0..NUM_SYMMETRIES {
		let c = get_symmetry_inv(cube, sym);
		let dst = to_idx(&c) as u32;

		if let Ok(k) = symtoraw.binary_search(&dst) {
			return Some((k, sym));
		}
	}

	None
}

fn gen_symmetrytoraw() -> SymToRawTable {
	create_symmetrytoraw_list(
		EDGE_ORI * UDSLICE,
		SYM_LEN,
		get_flipudslice_coord,
		cube_from_udslice_edge_idx,
	)
}

fn gen_symflipudslicetable(advanced_turns: bool) -> SymMovetable {
	create_sym_movetable(
		&toraw,
		cube_from_udslice_edge_idx,
		get_flipudslice_coord,
		get_turns_phase1(advanced_turns),
	)
}

fn gen_corner_ori_movetable(advanced_turns: bool) -> Movetable {
	create_movetable(
		CORNER_ORI,
		get_turns_phase1(advanced_turns),
		get_corner_ori_coord,
		cube_from_corner_ori,
	)
}

fn gen_corner_ori_symtable() -> Symtable {
	create_symtable(
		CORNER_ORI,
		16, // only use 16 symmetries
		get_corner_ori_coord,
		cube_from_corner_ori,
	)
}

fn gen_phase1_heuristics(advanced_turns: bool) -> Heuristics {
	let data_path = {
		let mut s = vec!["phase1", "heuristics"];
		if advanced_turns {
			s.push("adv_moves");
		}
		format!("data/{}.dat", s.join("-"))
	};

	if let Ok(data) = read_data::<35227103>(&data_path) {
		println!("Loaded {}", data_path);
		return Heuristics { data };
	}

	let ctable = gen_corner_ori_movetable(advanced_turns);
	let stable = gen_symflipudslicetable(advanced_turns);
	let symstate = gen_symstate(&toraw, get_flipudslice_coord, cube_from_udslice_edge_idx);

	let symtable = gen_corner_ori_symtable();

	if advanced_turns {
		gen_heuristics(data_path, symtable, ctable, symstate, stable, 10, 8)
	} else {
		gen_heuristics(data_path, symtable, ctable, symstate, stable, 12, 9)
	}
}

fn gen_symstate(
	symtoraw: &SymToRawTable,
	to_idx: fn(&CubieCube) -> usize,
	from_idx: fn(usize) -> CubieCube,
) -> SymState {
	(0..symtoraw.len())
		.into_par_iter()
		.map(|i| {
			let raw = symtoraw[i] as usize;
			let cube = from_idx(raw);

			// if a * sym[i] = a * sym[j], we have more
			// possible moves todo afterward!
			let mut out = 0;
			for sym in 1..16 {
				let csym = get_symmetry(&cube, sym);
				let idx = to_idx(&csym);

				if idx == raw {
					out |= 1 << sym;
				}
			}
			out as u32
		})
		.collect()
}

fn gen_heuristics(
	path: String,
	symtable: Symtable,
	movetable: Movetable,
	symstate: SymState,
	symmovetable: SymMovetable,
	max_depth: usize,
	rev_depth: usize,
) -> Heuristics {
	println!(
		"Must generate heuristics for {}, this may take a while...",
		path
	);

	let n = symmovetable.len() * movetable.len();

	let maxmoves = movetable.len();
	let numturns = movetable[0].len();

	const UNVISITED: u8 = u8::MAX;
	let mut out = vec![UNVISITED; n];
	out[0] = 0;

	for m in 0..(max_depth as u8) {
		#[cfg(debug_assertions)]
		let now = std::time::Instant::now();

		#[cfg(debug_assertions)]
		{
			let percent = m as f32 / (max_depth as f32) * 100f32;
			println!("Generating depth: {} ({}%)", m, percent);
		}

		if m <= rev_depth as u8 {
			// Forward generate
			let bound = if m == 0 { 1 } else { out.len() };
			for idx in 0..bound {
				if out[idx] != m {
					continue;
				}

				let twist0 = idx % maxmoves;
				let udslice0 = idx / maxmoves;

				for i in 0..numturns {
					let (sym_coord, sym) = symmovetable[udslice0][i];

					let twist1 = movetable[twist0][i] as usize;
					let twist = symtable[twist1][sym as usize];
					let dst = twist as usize + sym_coord as usize * maxmoves;

					if out[dst] != UNVISITED {
						continue;
					}
					out[dst] = m + 1;

					let state = symstate[sym_coord as usize];
					if state <= 1 {
						continue;
					}

					for j in 1..16 {
						if (state >> j) & 1 == 1 {
							let ddx = symtable[twist as usize][j as usize];
							let dst = ddx as usize + sym_coord as usize * maxmoves;

							if out[dst] != UNVISITED {
								continue;
							}
							out[dst] = m + 1;
						}
					}
				}
			}
		} else {
			// Backward search
			for idx in 0..n {
				if out[idx] != UNVISITED {
					continue;
				}

				let twist0 = idx % maxmoves;
				let sym0 = idx / maxmoves;

				for i in 0..numturns {
					let (sym_coord, sym) = symmovetable[sym0][i];

					let twist1 = movetable[twist0][i] as usize;
					let twist = symtable[twist1][sym as usize];
					let dst = twist as usize + sym_coord as usize * maxmoves;

					if out[dst] != m {
						continue;
					}
					out[idx] = m + 1;
					break;
				}
			}
		}

		#[cfg(debug_assertions)]
		println!("Time elapsed: {:.?}", now.elapsed());
	}

	#[cfg(debug_assertions)]
	{
		let mut cnt = vec![0; max_depth + 1];
		for x in out.iter() {
			cnt[*x as usize] += 1;
		}

		println!("Heuristics distribution:");
		for (i, tot) in cnt.iter().enumerate() {
			println!("{}: {}", i, tot);
		}
	}

	let h = Heuristics::from(out);

	match save_data(&path, &h.data) {
		Ok(()) => {}
		Err(_) => eprintln!("Could not save heuristics {}!", path),
	}

	h
}

// ===== Phase 2 =====

fn gen2_symmetrytoraw() -> SymToRawTable {
	create_symmetrytoraw_list(
		CORNER_PERM,
		SYM2_LEN,
		get_corner_perm_coord,
		cube_from_corner_perm,
	)
}

fn gen2_symmovetable(advanced_turns: bool) -> SymMovetable {
	create_sym_movetable(
		&toraw2,
		cube_from_corner_perm,
		get_corner_perm_coord,
		get_turns_phase2(advanced_turns),
	)
}

fn gen2_edge_perm_movetable(advanced_turns: bool) -> Movetable {
	create_movetable(
		EDGE8_PERM,
		get_turns_phase2(advanced_turns),
		get_edge8_perm,
		// we can actually do this because, the final 4 edges are never set
		// TODO Still, make this more understandable and cleaner and the same for gen2_edge_perm_symtable
		cube_from_edge_perm,
	)
}

fn gen2_edge_perm_symtable() -> Symtable {
	fn get_phase2_edge_perm(cube: &CubieCube) -> usize {
		cube.get_edge8_permutation_coord()
	}

	create_symtable(EDGE8_PERM, 16, get_phase2_edge_perm, cube_from_edge_perm)
}

/// Generate or load phase2 heuristics
fn gen_phase2_heuristics(advanced_turns: bool) -> Heuristics {
	let data_path = {
		let mut s = vec!["phase2", "heuristics"];
		if advanced_turns {
			s.push("adv_moves");
		}
		format!("data/{}.dat", s.join("-"))
	};

	if let Ok(data) = read_data::<27901440>(&data_path) {
		println!("Loaded {}", data_path);
		return Heuristics { data };
	}

	let stable = gen2_symmovetable(advanced_turns);
	let etable = gen2_edge_perm_movetable(advanced_turns);
	let symtable = gen2_edge_perm_symtable();

	let symstate = gen_symstate(&toraw2, get_corner_perm_coord, cube_from_corner_perm);

	if advanced_turns {
		gen_heuristics(data_path, symtable, etable, symstate, stable, 17, 11)
	} else {
		gen_heuristics(data_path, symtable, etable, symstate, stable, 18, 12)
	}
}

// ===== Solving =====

fn save_data(filepath: &str, data: &[u8]) -> Result<()> {
	let mut path = std::env::current_dir()?;
	path.push(filepath);

	let mut file = std::fs::File::create(path)?;
	file.write_all(data)?;
	Ok(())
}

fn read_data<const N: usize>(filepath: &str) -> Result<Vec<u8>> {
	let mut path = std::env::current_dir()?;
	path.push(filepath);

	let mut file = std::fs::File::open(path)?;
	let mut out = vec![0u8; N];
	file.read_exact(&mut out)?;

	Ok(out)
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
	get_slice_coord_sorted(cube, &udslice)
}

fn get_rlslice_sorted_coord(cube: &CubieCube) -> usize {
	get_slice_coord_sorted(cube, &rlslice)
}

fn get_fbslice_sorted_coord(cube: &CubieCube) -> usize {
	get_slice_coord_sorted(cube, &fbslice)
}

fn gen_toedge8perm_table() -> Movetable {
	(0..SLICE_SORTED)
		.into_par_iter()
		.map(|idx| {
			let mut cube = cube_from_fbslice_idx(idx);

			// Check if the udslice only contains edges from the udslice
			if udslice
				.iter()
				.any(|e| !udslice.contains(&cube.edges[*e as usize].0))
			{
				return vec![u16::MAX; 24];
			}

			// Get the indices where the RL-slice edges are
			let indices: Vec<_> = cube
				.edges
				.iter()
				.take(8)
				.enumerate()
				.filter(|(_, (e, _))| rlslice.contains(e))
				.map(|(i, _)| i)
				.collect();

			(0..24)
				.map(|j| {
					let rlslice_edges = permute_vec(rlslice.clone().to_vec(), j);

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

pub fn solve(initial: ArrayCube, advanced_turns: bool) -> Option<Vec<Turn>> {
	let cube: CubieCube = initial.try_into().unwrap();
	let mut solver = Solver::new(advanced_turns);
	solver.solve(cube)
}
