use std::io::{Read, Result, Write};

use lazy_static::lazy_static;
use strum::IntoEnumIterator;

use crate::{arraycube::*, cubiecube::{get_symmetry, get_symmetry_inv, CubieCube, Ori}, parse_turns, Corner, Edge, RubiksCube, Turn, NUM_CORNERS, NUM_EDGES, NUM_SIDES};
use crate::math::*;

/// v[coord][i] is the coordinate when applying move i on coord
type Movetable = Vec<Vec<u16>>;

/// v[coord][i] is the coordinate of the i-th symmetry of coord
type Symtable = Vec<Vec<u16>>;

/// v[symcoord][i] = (dst, sym) results by applying the move i on the symcoordinate "symcoord"
/// where dst: Is the resulting sym-coordinate
/// where sym: Is the sym-th symmetry of the to raw coordinate of symcoord
type SymMovetable = Vec<Vec<(u16,u8)>>;

lazy_static! {
	static ref cornersym: Symtable = gen_corner_ori_symtable();
	static ref edgesym: Symtable = gen2_edge_perm_symtable();

	static ref toraw: Vec<u32> = gen_symmetrytoraw();
	static ref toraw2: Vec<u32> = gen2_symmetrytoraw();

	static ref h1: Vec<u8> = gen_phase1_heuristics();
	static ref h2: Vec<u8> = gen_phase2_heuristics();

	static ref turns_phase1: Vec<Turn> = parse_turns("U U2 U' D D2 D' B B2 B' F F2 F' L L2 L' R R2 R'").unwrap();
	static ref turns_phase2: Vec<Turn> = parse_turns("U U2 U' D D2 D' B2 F2 L2 R2").unwrap();

	static ref rlslice: Vec<Edge> = vec![Edge::UF, Edge::DF, Edge::DB, Edge::UB];
	static ref fbslice: Vec<Edge> = vec![Edge::UR, Edge::DR, Edge::DL, Edge::UL];
	static ref udslice: Vec<Edge> = vec![Edge::FR, Edge::BR, Edge::BL, Edge::FL];
}

// Helper list to map the index of the edge in a slice
const EDGE_SLICE_INDEX: [usize; NUM_EDGES] = [
	0, 0, 3, 3, 1, 1, 2, 2, 0, 1, 2, 3,
];

const CORNER_ORI: usize = 2187;
const EDGE_ORI: usize = 2048;
const SYM_LEN: usize = 64430;
const UDSLICE: usize = 495;

const CORNER_PERM: usize = 40320;
const EDGE8_PERM: usize = 40320;
const SYM2_LEN: usize = 2768;

fn get_flipudslice_coord(cube: &CubieCube) -> usize {
	let edge = cube.get_edge_orientation_coord();
	let udslice_coord = cube.get_udslice_coord();
	udslice_coord * EDGE_ORI + edge
}

fn cube_from_edge_ori_idx(idx: usize) -> CubieCube {
	let mut cube = CubieCube::new();

	for i in 0..(NUM_EDGES-1) {
		cube.edges[i].1 = (idx >> i) as Ori & 1;
	}
	cube.edges[NUM_EDGES-1].1 = idx.count_ones() % 2 as Ori;

	cube
}

fn cube_from_corner_ori_idx(idx: usize) -> CubieCube {
	let mut x = idx;
	let mut parity = 0;
	let mut cube = CubieCube::new();

	for corner in Corner::iter().take(NUM_CORNERS-1) {
		cube.corners[corner as usize] = (corner, x as Ori % 3);
		parity = (parity + x) % 3;
		x /= 3;
	}
	cube.corners[NUM_CORNERS-1].1 = (3-parity) as Ori % 3;

	cube
}

fn cube_from_edge_pos_idx(edges: Vec<Edge>, idx: usize) -> CubieCube {
	let mut cube = CubieCube::new();

	let chosen = get_nck(NUM_EDGES, edges.len(), idx);

	let mut m = 0;
	let mut n = 0;
	let nonmiddle: Vec<_> = Edge::iter().filter(|e| !edges.contains(&e)).collect();

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

	let mut cube = cube_from_edge_ori_idx(edge_ori_coord);
	let edge_cube = cube_from_edge_pos_idx(udslice.to_vec(), udslice_coord);
	for i in 0..NUM_EDGES {
		cube.edges[i].0 = edge_cube.edges[i].0;
	}

	cube
}

fn permute_vec<T>(v: Vec<T>, k: usize) -> Vec<T>
where T: Clone {
	get_kth_perm(v.len(), k).into_iter().map(|i| {
		v[i].clone()
	}).collect()
}

fn cube_from_rlslice_idx(idx: usize) -> CubieCube {
	let pos = idx / 24;
	let ord = idx % 24;

	let slice = permute_vec(rlslice.to_vec(), ord);
	cube_from_edge_pos_idx(slice, pos)
}

fn cube_from_fbslice_idx(idx: usize) -> CubieCube {
	let pos = idx / 24;
	let ord = idx % 24;

	let slice = permute_vec(fbslice.to_vec(), ord);
	cube_from_edge_pos_idx(slice, pos)
}

fn cube_from_corner_perm(idx: usize) -> CubieCube {
	let mut cube = CubieCube::new();

	let cs: Vec<Corner> = permute_vec(Corner::iter().collect(), idx);
	for (i, corner) in cs.into_iter().enumerate() {
		cube.corners[i].0 = corner;
	}

	cube
}

fn cube_from_edge_perm(idx: usize, n: usize) -> CubieCube {
	let mut cube = CubieCube::new();

	let cs: Vec<Edge> = permute_vec(Edge::iter().take(n).collect(), idx);
	for (i, edge) in cs.into_iter().enumerate() {
		cube.edges[i].0 = edge;
	}

	cube
}

// ===== Table Generating =====

/// Create a movetable
fn create_movetable(
	num_states: usize,
	moves: &Vec<Turn>,
	to_idx: fn(&CubieCube) -> usize,
	from_idx: fn(usize) -> CubieCube,
) -> Movetable {
	let mut out = vec![ vec![0; moves.len()]; num_states ];

	for idx in 0..num_states {
		// Create cube from current idx
		let cube = from_idx(idx);

		// Apply each turn to the cube and save the corresponding index
		for (i, turn) in moves.iter().enumerate() {
			let ncube = {
				let mut ncube = cube.clone();
				ncube.apply_turn(*turn);
				ncube
			};

			out[idx][i] = to_idx(&ncube) as u16;
		}
	}

	out
}

fn create_symmetrytoraw_list(
	num_states: usize,
	num_symmetry_states: usize,
	to_idx: fn(&CubieCube) -> usize,
	from_idx: fn(usize) -> CubieCube,
) -> Vec<u32> {
	let mut out = vec![0u32; num_symmetry_states];
	let mut used = bit_set::BitSet::with_capacity( num_states );
	let mut symidx = 0;

	for idx in 0..num_states {
		if used.contains(idx) { continue; }

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

	#[cfg(debug_assertions)]
	{
		for i in 0..num_symmetry_states-1 {
			assert!(out[i] < out[i+1]);
		}
	}

	out
}

fn create_symtable(
	num_states: usize,
	num_symmetries: usize,
	to_idx: fn(&CubieCube) -> usize,
	from_idx: fn(usize) -> CubieCube,
) -> Symtable {
	let mut out = vec![vec![0; num_symmetries]; SYM_LEN];

	for idx in 0..num_states {
		let cube = from_idx(idx);

		for sym in 0..num_symmetries {
			let csym = get_symmetry_inv(&cube, sym);
			out[idx][sym] = to_idx(&csym) as u16;
		}
	}

	out
}

fn create_sym_movetable(
	symtoraw: &Vec<u32>,
	from_idx: fn(usize) -> CubieCube,
	to_idx: fn(&CubieCube) -> usize,
	moves: &Vec<Turn>,
) -> SymMovetable {
	let n = symtoraw.len();
	let mut out = vec![ vec![(0, 0); moves.len()]; n ];

	for idx in 0..n {
		// Create cube from current symtoraw[idx]
		let cube = from_idx( symtoraw[idx] as usize );

		// Apply each turn to the cube and save the corresponding index
		for (i, turn) in moves.iter().enumerate() {
			let ncube = {
				let mut ncube = cube.clone();
				ncube.apply_turn(*turn);
				ncube
			};

			let (dst, sym) = get_sym_class(&ncube, symtoraw, to_idx).unwrap();
			out[idx][i] = (dst as u16, sym as u8);
		}
	}

	out
}

fn get_sym_class(
	cube: &CubieCube,
	symtoraw: &Vec<u32>,
	to_idx: fn(&CubieCube) -> usize,
) -> Option<(usize,usize)> {
	for sym in 0..48 {
		let c = get_symmetry_inv(&cube, sym);
		let dst = to_idx(&c) as u32;

		if let Ok(k) = symtoraw.binary_search(&dst) {
			return Some((k, sym));
		}
	}

	None
}

fn gen_symmetrytoraw() -> Vec<u32> {
	create_symmetrytoraw_list(
		EDGE_ORI*UDSLICE,
		SYM_LEN,
		get_flipudslice_coord,
		cube_from_udslice_edge_idx
	)
}


fn gen_symflipudslicetable() -> SymMovetable {
	create_sym_movetable(
		&toraw,
		cube_from_udslice_edge_idx,
		get_flipudslice_coord,
		&turns_phase1
	)
}



fn gen_corner_ori_movetable() -> Movetable {
	fn get_corner_ori_idx(cube: &CubieCube) -> usize {
		cube.get_corner_orientation_coord()
	}

	create_movetable(
		CORNER_ORI,
		&turns_phase1,
		get_corner_ori_idx,
		cube_from_corner_ori_idx
	)
}

fn gen_corner_ori_symtable() -> Symtable {
	fn get_corner_ori_idx(cube: &CubieCube) -> usize {
		cube.get_corner_orientation_coord()
	}

	create_symtable(
		CORNER_ORI,
		16,
		get_corner_ori_idx,
		cube_from_corner_ori_idx,
	)
}


fn gen_phase1_heuristics() -> Vec<u8> {
	const DATA_PATH: &str = "data/heuristics.dat";

	if let Ok(h) = read_data::<{SYM_LEN*CORNER_ORI}>(DATA_PATH) {
		return h.into();
	}

	println!("Must generate heuristics for phase 1, please wait...");

	let ctable = gen_corner_ori_movetable();
	let stable = gen_symflipudslicetable();

	// Used to analyze symmetries of stable
	let symstate: Vec<u16> = {
		let mut out = vec![0; SYM_LEN];

		for i in 0..SYM_LEN {
			let raw = toraw[i] as usize;
			let cube = cube_from_udslice_edge_idx(raw);

			for sym in 1..16 {
				let csym = get_symmetry(&cube, sym);
				let idx = get_flipudslice_coord(&csym);

				if idx == raw {
					out[i] |= 1 << sym;
				}
			}
		}

		out
	};

	const UNVISITED: u8 = u8::MAX;
	let mut out = vec![UNVISITED; SYM_LEN*CORNER_ORI];
	out[0] = 0;

	for m in 0..12 {
		if m <= 8 { // Forward generate
			for idx in 0..SYM_LEN*CORNER_ORI {
				if out[idx] != m { continue; }

				let twist0 = idx % CORNER_ORI;
				let udslice0 = idx / CORNER_ORI;

				for i in 0..turns_phase1.len() {
					let (udslice_coord, sym) = stable[udslice0 as usize][i];

					let twist1 = ctable[twist0 as usize][i] as usize;
					let twist = cornersym[twist1][sym as usize];
					let dst = twist as usize + udslice_coord as usize * CORNER_ORI;

					if out[dst] != UNVISITED { continue; }
					out[dst] = m+1;

					let state = symstate[udslice_coord as usize];
					if state == 1 { continue; }

					for j in 1..16 {
						if (state >> j) & 1 == 1 {
							let ddx = cornersym[twist as usize][j as usize];
							let dst = ddx as usize + udslice_coord as usize * CORNER_ORI;

							if out[dst] != UNVISITED { continue; }
							out[dst] = m+1;
						}
					}
				}
			}
		} else { // Backward search
			for idx in 0..SYM_LEN*CORNER_ORI {
				if out[idx] != UNVISITED { continue; }

				let twist0 = idx % CORNER_ORI;
				let udslice0 = idx / CORNER_ORI;

				for i in 0..turns_phase1.len() {
					let (udslice_coord, sym) = stable[udslice0 as usize][i];

					let twist1 = ctable[twist0 as usize][i] as usize;
					let twist = cornersym[twist1][sym as usize];
					let dst = twist as usize + udslice_coord as usize * CORNER_ORI;

					if out[dst] != m { continue; }
					out[idx] = m+1;
					break;
				}
			}
		}
	}

	match save_data(DATA_PATH, &out) {
		Ok(()) => {},
		Err(_) => eprintln!("Could not save heuristics for phase 1!"),
	}

	out
}

// ===== Phase 2 =====

fn gen2_symmetrytoraw() -> Vec<u32> {
	fn get_corner_perm(cube: &CubieCube) -> usize {
		cube.get_corner_perm_coord()
	}

	create_symmetrytoraw_list(
		CORNER_PERM,
		SYM2_LEN,
		get_corner_perm,
		cube_from_corner_perm,
	)
}

/// Tables for phase 2 ===


fn gen2_symmovetable() -> SymMovetable {
	fn get_corn_perm(cube: &CubieCube) -> usize {
		cube.get_corner_perm_coord()
	}

	create_sym_movetable(
		&toraw2,
		cube_from_corner_perm,
		get_corn_perm,
		&turns_phase2
	)
}


fn gen2_edge_perm_movetable() -> Movetable {
	fn get_phase2_edge_perm(cube: &CubieCube) -> usize {
		cube.get_edge8_perm_coord()
	}

	fn from_coord(idx: usize) -> CubieCube {
		cube_from_edge_perm(idx, 8)
	}

	create_movetable(
		EDGE8_PERM,
		&turns_phase2,
		get_phase2_edge_perm,
		from_coord,
	)
}

fn gen2_edge_perm_symtable() -> Symtable {
	fn get_phase2_edge_perm(cube: &CubieCube) -> usize {
		cube.get_edge8_perm_coord()
	}

	fn from_coord(idx: usize) -> CubieCube {
		cube_from_edge_perm(idx, 8)
	}

	create_symtable(
		EDGE8_PERM,
		16,
		get_phase2_edge_perm,
		from_coord,
	)
}


fn gen_phase2_heuristics() -> Vec<u8> {
	const DATA_PATH: &str = "data/heuristics_phase2.dat";
	if let Ok(data) = read_data::<{SYM2_LEN*EDGE8_PERM}>(DATA_PATH) {
		return data.into();
	}
	println!("Must generate heuristics for phase 2, please wait...");

	let stable = gen2_symmovetable();
	let etable = gen2_edge_perm_movetable();

	const UNVISITED: u8 = u8::MAX;
	let mut out = vec![UNVISITED; SYM2_LEN*EDGE8_PERM];
	out[0] = 0;

	let symstate: Vec<u16> = {
		let mut out = vec![1u16; SYM2_LEN];

		for i in 0..SYM2_LEN {
			let raw = toraw2[i] as usize;
			let cube = cube_from_corner_perm(raw);

			for sym in 1..16 {
				let csym = get_symmetry(&cube, sym);
				let idx = csym.get_corner_perm_coord();

				if idx == raw {
					out[i] |= 1 << sym;
				}
			}
		}

		out
	};

	for m in 0..=18 {
		if m <= 12 { // Forward generate
			for idx in 0..SYM2_LEN*EDGE8_PERM {
				if out[idx] != m { continue; }

				let edge0 = idx / SYM2_LEN;
				let corner0 = idx % SYM2_LEN;

				for i in 0..10 {
					let (corner, sym) = stable[corner0 as usize][i];

					let edge1 = etable[edge0 as usize][i] as usize;
					let edge = edgesym[edge1][sym as usize];
					let dst = edge as usize * SYM2_LEN + corner as usize;

					if out[dst] != UNVISITED { continue; }
					out[dst] = m+1;

					let state = symstate[corner as usize];
					if state == 1 { continue; }

					for j in 1..16 {
						if (state >> j) & 1 == 1 {
							let ddx = edgesym[edge as usize][j as usize];
							let dst = ddx as usize * SYM2_LEN + corner as usize;

							if out[dst] != UNVISITED { continue; }
							out[dst] = m+1;
						}
					}
				}
			}
		} else { // Backward search
			for idx in 0..SYM2_LEN*EDGE8_PERM {
				if out[idx] != UNVISITED { continue; }

				let edge0 = idx / SYM2_LEN;
				let corner0 = idx % SYM2_LEN;

				for i in 0..turns_phase2.len() {
					let (corner, sym) = stable[corner0 as usize][i];

					let edge1 = etable[edge0 as usize][i] as usize;
					let edge = edgesym[edge1][sym as usize];
					let dst = edge as usize * SYM2_LEN + corner as usize;

					if out[dst] != m { continue; }
					out[idx] = m+1;
					break;
				}
			}
		}
	}

	match save_data(DATA_PATH, &out) {
		Ok(()) => {},
		Err(_) => eprintln!("Could not save heuristics for phase 2!"),
	}

	out
}

fn get_phase2_coord(cube: &CubieCube) -> usize {
	fn corner_perm(cube: &CubieCube) -> usize { cube.get_corner_perm_coord() }

	let (z, sym) = get_sym_class(&cube, &toraw2, corner_perm).unwrap();
	let edge = cube.get_edge8_perm_coord();

	let y = edgesym[ edge ][sym] as usize;
	y*SYM2_LEN + z
}

// ===== Solving =====

fn save_data(filepath: &str, data: &Vec<u8>) -> Result<()> {
	let mut path = std::env::current_dir()?;
	path.push(filepath);

	let mut file = std::fs::File::create(path)?;
	file.write(data)?;
	Ok(())
}

fn read_data<const N: usize>(filepath: &str) -> Result<Vec<u8>> {
	let mut path = std::env::current_dir()?;
	path.push(filepath);

	let mut file = std::fs::File::open(path)?;
	let mut out = vec![0u8; N];
	file.read(&mut out)?;

	Ok(out)
}

fn search_phase2(path: &mut Vec<Turn>, cube: CubieCube, g: usize, bound: usize) -> usize {
	let f = g + h2[ get_phase2_coord(&cube) ] as usize;
	if f > bound { return f; }

	if cube.is_solved() { return usize::MAX-1; }

	let mut min = usize::MAX;

	let mut ord: Vec<(u8,usize)> = vec![];
	for (i, turn) in turns_phase2.iter().enumerate() {
		let dst = {
			let mut nc = cube.clone();
			nc.apply_turn(*turn);
			get_phase2_coord(&nc)
		};

		ord.push( (h2[dst], i) )
	}

	ord.sort();

	for turn in ord.iter().map(|(_,i)| turns_phase2[*i]) {
		let ncube = {
			let mut nc = cube.clone();
			nc.apply_turn(turn);
			nc
		};

		path.push(turn);

		let t = search_phase2(path, ncube, g+1, bound);
		if t == usize::MAX-1 { return usize::MAX-1; }
		min = std::cmp::min(min, t);

		path.pop();
	}

	min
}

const SLICE_COORD: usize = 495*24;

fn gen_rlslice_sorted() -> Movetable {
	fn get_rlslice_coord(cube: &CubieCube) -> usize {
		let vec = cube.edges.iter().map(|(e,_)| rlslice.contains(&e)).collect();
		let ord = cube.edges.iter()
			.filter(|(e,_)| rlslice.contains(&e))
			.map(|(e,_)| EDGE_SLICE_INDEX[*e as usize])
			.collect();
		map_nck(&vec) * 24 + map_permutation(&ord)
	}

	create_movetable(
		SLICE_COORD,
		&turns_phase1,
		get_rlslice_coord,
		cube_from_rlslice_idx
	)
}


fn gen_fbslice_sorted() -> Movetable {
	fn get_fbslice_coord(cube: &CubieCube) -> usize {
		let vec = cube.edges.iter().map(|(e,_)| fbslice.contains(&e)).collect();
		let ord = cube.edges.iter()
			.filter(|(e,_)| fbslice.contains(&e))
			.map(|(e,_)| EDGE_SLICE_INDEX[*e as usize])
			.collect();
		map_nck(&vec) * 24 + map_permutation(&ord)
	}

	create_movetable(
		SLICE_COORD,
		&turns_phase1,
		get_fbslice_coord,
		cube_from_fbslice_idx
	)
}

fn gen_toedge8perm_table() -> Movetable {
	let mut out = vec![ vec![u16::MAX; 24]; SLICE_COORD ];

	for idx in 0..SLICE_COORD {
		let mut cube = cube_from_fbslice_idx(idx);
		// Check if the udslice only contains edges from the udslice
		if udslice.iter().any(|e| !udslice.contains(&cube.edges[*e as usize].0)) {
			continue;
		}

		// Get the indices where the RL-slice edges are
		let indices: Vec<_> = Edge::iter()
			.take(8)
			.enumerate()
			.filter(|(_,e)| rlslice.contains(e))
			.map(|(i,_)| i)
			.collect();

		for j in 0..24 {
			let rlslice_edges = permute_vec(rlslice.to_vec(), j);

			for (i, ele) in indices.iter().enumerate() {
				cube.edges[*ele].0 = rlslice_edges[i];
			}

			out[idx][j] = cube.get_corner_perm_coord() as u16;
		}
	}

	out
}

pub fn solve(initial: ArrayCube) -> Option<Vec<Turn>> {
	let mut turns = vec![];
	let mut cube: CubieCube = initial.clone().into();

	let mut dist = {
		let (z, sym) = get_sym_class(&cube, &toraw, get_flipudslice_coord).unwrap();
		let y = cornersym[ cube.get_corner_orientation_coord() ][sym] as usize;
		h1[z*CORNER_ORI + y]
	};

	loop {
		if dist == 0 { break; }

		for turn in turns_phase1.iter() {
			let mut dst = cube.clone();
			dst.apply_turn(*turn);

			let (dz, sym) = get_sym_class(&dst, &toraw, get_flipudslice_coord).unwrap();
			let dy = cornersym[ dst.get_corner_orientation_coord() ][sym] as usize;

			let ddist = h1[ dy + dz*CORNER_ORI ];

			if ddist+1 == dist {
				cube.apply_turn(*turn);
				turns.push(*turn);
				dist = ddist;
				break;
			}
		}
	}

	// phase 2

	let mut bound = h2[ get_phase2_coord(&cube) ] as usize;
	let mut path = vec![];

	loop {
		let t = search_phase2(&mut path, cube.clone(), 0, bound);
		if t == usize::MAX-1 {
			for p in path { turns.push(p); }
			break;
		}
		if t == usize::MAX { return None; }
		bound = t;
	}

	#[cfg(debug_assertions)]
	{
		let mut c = initial.clone();
		for turn in turns.iter() {
			c.apply_turn(*turn);
		}
		assert!(c.is_solved());
	}

	Some(turns)
}
