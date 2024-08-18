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

type SymState = Vec<u32>;

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
	static ref edgesym: Symtable = gen2_edge_perm_symtable();
	static ref cperm_symtable: Symtable = create_symtable(
		CORNER_PERM,
		16,
		get_corner_perm_coord,
		cube_from_corner_perm
	);
}

// Helper list to map the index of the edge in a slice
const EDGE_SLICE_INDEX: [usize; NUM_EDGES] = [0, 0, 3, 3, 1, 1, 2, 2, 0, 1, 2, 3];

const SYM_LEN: usize = 64430;
const UDSLICE: usize = 495;

const EDGE8_PERM: usize = 40320;
const SYM2_LEN: usize = 2768;

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

fn get_udslice_coord(cube: &CubieCube) -> usize {
	cube.get_udslice_coord()
}

fn cube_from_edge_ori_idx(coord: usize) -> CubieCube {
	let mut cube = CubieCube::new();
	cube.set_edge_orientation(coord);
	cube
}

fn cube_from_corner_ori_idx(coord: usize) -> CubieCube {
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

	let mut cube = cube_from_edge_ori_idx(edge_ori_coord);
	let edge_cube = cube_from_edge_pos_idx(udslice.to_vec(), udslice_coord);
	for i in 0..NUM_EDGES {
		cube.edges[i].0 = edge_cube.edges[i].0;
	}

	cube
}

fn cube_from_udslice_coord(idx: usize) -> CubieCube {
	let mut cube = CubieCube::new();

	let edge_cube = cube_from_edge_pos_idx(udslice.to_vec(), idx);
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
					let ncube = {
						let mut ncube = cube.clone();
						ncube.apply_turn(*turn);
						ncube
					};

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

	#[cfg(debug_assertions)]
	{
		for i in 0..num_symmetry_states - 1 {
			assert!(out[i] < out[i + 1]);
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
		cube_from_corner_ori_idx,
	)
}

fn gen_corner_ori_symtable() -> Symtable {
	create_symtable(
		CORNER_ORI,
		16, // only use 16 symmetries
		get_corner_ori_coord,
		cube_from_corner_ori_idx,
	)
}

fn gen_phase1_heuristics(advanced_turns: bool) -> Vec<u8> {
	let data_path = {
		let mut s = vec!["phase1", "heuristics"];
		if advanced_turns {
			s.push("adv_moves");
		}
		format!("data/{}.dat", s.join("-"))
	};

	if let Ok(h) = read_data::<{ SYM_LEN * CORNER_ORI }>(&data_path) {
		return h;
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
	let n = symtoraw.len();

	(0..n)
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
) -> Vec<u8> {
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
			let percent = m as f32 / 12f32 * 100f32;
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

	match save_data(&path, &out) {
		Ok(()) => {}
		Err(_) => eprintln!("Could not save heuristics {}!", path),
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

	out
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
	fn get_phase2_edge_perm(cube: &CubieCube) -> usize {
		cube.get_edge8_permutation_coord()
	}

	create_movetable(
		EDGE8_PERM,
		get_turns_phase2(advanced_turns),
		get_phase2_edge_perm,
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
fn gen_phase2_heuristics(advanced_turns: bool) -> Vec<u8> {
	let data_path = {
		let mut s = vec!["phase2", "heuristics"];
		if advanced_turns {
			s.push("adv_moves");
		}
		format!("data/{}.dat", s.join("-"))
	};

	if let Ok(data) = read_data::<{ SYM2_LEN * EDGE8_PERM }>(&data_path) {
		return data;
	}

	let stable = gen2_symmovetable(advanced_turns);
	let etable = gen2_edge_perm_movetable(advanced_turns);
	let symtable = gen2_edge_perm_symtable();

	const UNVISITED: u8 = u8::MAX;
	let mut out = vec![UNVISITED; SYM2_LEN * EDGE8_PERM];
	out[0] = 0;

	let symstate = gen_symstate(&toraw2, get_corner_perm_coord, cube_from_corner_perm);

	if advanced_turns {
		gen_heuristics(data_path, symtable, etable, symstate, stable, 17, 11)
	} else {
		gen_heuristics(data_path, symtable, etable, symstate, stable, 18, 12)
	}
}

fn get_phase2_coord2(corner_perm: usize, edge8_perm: usize) -> usize {
	let mut out = (0, 0);
	for sym in 0..16 {
		let dx = cperm_symtable[corner_perm][sym] as u32;

		if let Ok(k) = toraw2.binary_search(&dx) {
			out = (k, sym);
			break;
		}
	}

	let (x, sym) = out;
	let y = edgesym[edge8_perm][sym] as usize;

	y + x * EDGE8_PERM
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

#[allow(clippy::too_many_arguments)]
fn search_phase2(
	h2: &Vec<u8>,
	corner_perm: usize,
	corner_perm_table: &Movetable,
	edge8_perm: usize,
	edge8_perm_table: &Movetable,
	udslice_coord: usize,
	udslice_movetable: &Movetable,

	turns: &Vec<Turn>,
	path: &mut Vec<Turn>,
	g: usize,
	bound: usize,
) -> usize {
	let f = g + h2[get_phase2_coord2(corner_perm, edge8_perm)] as usize;
	if f > bound {
		return f;
	}

	if corner_perm == 0 && edge8_perm == 0 && udslice_coord == 0 {
		return usize::MAX - 1;
	}

	let mut min = usize::MAX;

	let mut ord: Vec<(u8, usize)> = turns
		.iter()
		.enumerate()
		.map(|(i, _)| {
			let d_cperm = corner_perm_table[corner_perm][i] as usize;
			let d_eperm = edge8_perm_table[edge8_perm][i] as usize;

			let dst = get_phase2_coord2(d_cperm, d_eperm);
			(h2[dst], i)
		})
		.collect();

	ord.sort();

	for (_, i) in ord {
		let d_cperm = corner_perm_table[corner_perm][i] as usize;
		let d_eperm = edge8_perm_table[edge8_perm][i] as usize;
		let d_udslice = udslice_movetable[udslice_coord][i] as usize;

		path.push(turns[i]);

		// let t = search_phase2(h2, turns, path, ncube, g + 1, bound);
		let t = search_phase2(
			h2,
			d_cperm,
			corner_perm_table,
			d_eperm,
			edge8_perm_table,
			d_udslice,
			udslice_movetable,
			turns,
			path,
			g + 1,
			bound,
		);
		if t == usize::MAX - 1 {
			return usize::MAX - 1;
		}
		min = std::cmp::min(min, t);

		path.pop();
	}

	min
}

const SLICE_COORD: usize = 495 * 24;

fn get_slice_coord_sorted(cube: &CubieCube, slice: &[Edge]) -> usize {
	let vec: Vec<_> = cube
		.edges
		.iter()
		.map(|(e, _)| slice.contains(e))
		.collect();
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

fn gen_rlslice_sorted(advanced_turns: bool) -> Movetable {
	create_movetable(
		SLICE_COORD,
		get_turns_phase1(advanced_turns),
		get_rlslice_sorted_coord,
		cube_from_rlslice_idx,
	)
}

fn gen_fbslice_sorted(advanced_turns: bool) -> Movetable {
	create_movetable(
		SLICE_COORD,
		get_turns_phase1(advanced_turns),
		get_fbslice_sorted_coord,
		cube_from_fbslice_idx,
	)
}

fn gen_toedge8perm_table() -> Movetable {
	(0..SLICE_COORD)
		.into_par_iter()
		.map(|idx| {
			let mut cube = cube_from_fbslice_idx(idx);
			// Check if the udslice only contains edges from the udslice
			if udslice
				.par_iter()
				.any(|e| !udslice.contains(&cube.edges[*e as usize].0))
			{
				return vec![u16::MAX; 24];
			}

			// Get the indices where the RL-slice edges are
			let indices: Vec<_> = Edge::iter()
				.take(8)
				.enumerate()
				.filter(|(_, e)| rlslice.contains(e))
				.map(|(i, _)| i)
				.collect();

			(0..24)
				.map(|j| {
					let rlslice_edges = permute_vec(rlslice.to_vec(), j);

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

#[allow(clippy::too_many_arguments)]
fn phase2(
	h2: &Vec<u8>,
	corner_perm: usize,
	corner_perm_table: &Movetable,
	edge8_perm: usize,
	edge8_perm_table: &Movetable,
	udslice_coord: usize,
	udslice_movetable: &Movetable,
	advanced_turns: bool,
) -> Option<Vec<Turn>> {
	let idx = get_phase2_coord2(corner_perm, edge8_perm);
	let mut bound = h2[idx] as usize;
	let mut path = vec![];
	let turns_phase2 = get_turns_phase2(advanced_turns);

	loop {
		let t = search_phase2(
			h2,
			corner_perm,
			corner_perm_table,
			edge8_perm,
			edge8_perm_table,
			udslice_coord,
			udslice_movetable,
			&turns_phase2,
			&mut path,
			0,
			bound,
		);
		if t == usize::MAX - 1 {
			break;
		}
		if t == usize::MAX {
			return None;
		}
		bound = t;
	}

	Some(path)
}

/// Solves the Rubik's Cube using Kociemba's Algorithm
pub fn solve(initial: ArrayCube, advanced_turns: bool) -> Option<Vec<Turn>> {
	let mut turns = vec![];
	let cube: CubieCube = {
		let res = initial.clone().try_into();

		match res {
			Ok(cube) => cube,
			Err(e) => {
				eprintln!("Cube can't be solved: {}", e);
				return None;
			}
		}
	};

	let cornersym = gen_corner_ori_symtable();

	let now = std::time::Instant::now();
	let h1 = gen_phase1_heuristics(advanced_turns);
	println!("Time elapsed for H1: {:.?}", now.elapsed());

	let turns_phase1 = get_turns_phase1(advanced_turns);

	let fbslice_table = gen_fbslice_sorted(advanced_turns);
	let rlslice_table = gen_rlslice_sorted(advanced_turns);

	let udslice_movetable = create_movetable(
		SLICE_COORD,
		turns_phase1.clone(),
		get_udslice_sorted_coord,
		cube_from_udslice_sorted,
	);
	let edge_movetable = create_movetable(
		EDGE_ORI,
		turns_phase1.clone(),
		get_edge_ori_coord,
		cube_from_edge_ori_idx,
	);
	let corner_movetable = gen_corner_ori_movetable(advanced_turns);

	let udslice_symtable = create_symtable(UDSLICE, 16, get_udslice_coord, cube_from_udslice_coord);
	let edge_symtable = create_symtable(EDGE_ORI, 16, get_edge_ori_coord, cube_from_edge_ori_idx);

	let mut udslice_sorted_coord = get_udslice_sorted_coord(&cube);
	let mut rlslice_sorted_coord = get_rlslice_sorted_coord(&cube);
	let mut fbslice_sorted_coord = get_fbslice_sorted_coord(&cube);

	let mut edge_ori = cube.get_edge_orientation_coord();
	let mut corner_ori = cube.get_corner_orientation_coord();

	let mut corner_perm = cube.get_corner_perm_coord();

	let corner_table1 = create_movetable(
		CORNER_PERM,
		turns_phase1.clone(),
		get_corner_perm_coord,
		cube_from_corner_perm,
	);
	let corner_table2 = create_movetable(
		CORNER_PERM,
		get_turns_phase2(advanced_turns),
		get_corner_perm_coord,
		cube_from_corner_perm,
	);

	let mut dist = {
		let mut coord = (0, 0);
		for sym in 0..16 {
			let usym = udslice_symtable[udslice_sorted_coord / 24][sym];
			let esym = edge_symtable[edge_ori][sym];

			let tcoord = (usym as usize * EDGE_ORI + esym as usize) as u32;
			if let Ok(k) = toraw.binary_search(&tcoord) {
				coord = (k, sym);
				break;
			}
		}

		let (dz, sym) = coord;
		let dy = cornersym[corner_ori][sym] as usize;
		h1[dy + dz * CORNER_ORI]
	};

	initial.print();
	println!("initial: {} {} {}", corner_ori, edge_ori, udslice_sorted_coord);
	println!("PHASE 1");

	let mut ccube = cube.clone();

	loop {
		if udslice_sorted_coord/24 == 0 && edge_ori == 0 && corner_ori == 0 {
			break;
		}

		let mut good = false;

		for (i, turn) in turns_phase1.iter().enumerate() {
			let d_udslice = udslice_movetable[udslice_sorted_coord][i] as usize;
			let d_edge_ori = edge_movetable[edge_ori][i] as usize;
			let d_corner_ori = corner_movetable[corner_ori][i] as usize;

			let mut coord = (0, 0);
			for sym in 0..16 {
				let usym = udslice_symtable[d_udslice / 24][sym] as usize;
				let esym = edge_symtable[d_edge_ori][sym] as usize;

				let tcoord = (usym * EDGE_ORI + esym) as u32;
				println!("COORD[{}]: {}", sym, tcoord);
				println!("{} {}", usym, esym);
				if let Ok(k) = toraw.binary_search(&tcoord) {
					coord = (k, sym);
					break;
				}
			}

			let (dz, sym) = coord;
			let dy = cornersym[d_corner_ori][sym] as usize;
			let ddist = h1[dy + dz * CORNER_ORI];

			#[cfg(debug_assertions)]
			{
				let mut nc = ccube.clone();
				nc.apply_turn(*turn);

				let mut out = (0,0);
				for sym in 0..NUM_SYMMETRIES {
					let c = get_symmetry_inv(&nc, sym);
					let dst = get_flipudslice_coord(&c) as u32;

					let esym = c.get_edge_orientation_coord();
					let usym = c.get_udslice_coord();

					println!("CUBE[{}]: {}", sym, dst);
					println!("{} {}", usym, esym);
					if let Ok(k) = toraw.binary_search(&dst) {
						out = (k, sym);
						break;
					}
				}

				let (ddz, dsym) = out;
				let ddy = cornersym[nc.get_corner_orientation_coord()][dsym] as usize;

				println!("{} {} {}", dz, sym, dy);
				println!("{} {} {}", ddz, dsym, ddy);

				assert_eq!(ddz, dz);
				assert_eq!(dsym, sym);
				assert_eq!(ddy, dy);
			}

			if ddist + 1 == dist {
				edge_ori = d_edge_ori;
				corner_ori = d_corner_ori;
				udslice_sorted_coord = d_udslice;

				rlslice_sorted_coord = rlslice_table[rlslice_sorted_coord][i] as usize;
				fbslice_sorted_coord = fbslice_table[fbslice_sorted_coord][i] as usize;
				corner_perm = corner_table1[corner_perm][i] as usize;

				ccube.apply_turn(*turn);
				turns.push(*turn);
				dist = ddist;
				good = true;
				break;
			}
		}
		if !good {
			panic!("failed: {} {} {}", corner_ori, edge_ori, udslice_sorted_coord);
		}
	}

	println!("GENERATE PHASE 2 STUFF");
	for turn in turns.iter() {
		print!("{} ", turn);
	}
	println!();

	let h2 = gen_phase2_heuristics(advanced_turns);

	let toedge8_perm = gen_toedge8perm_table();
	let edge8_perm = toedge8_perm[fbslice_sorted_coord][rlslice_sorted_coord % 24] as usize;
	let edge8_perm_table = gen2_edge_perm_movetable(advanced_turns);

	let udslice_movetable2 = create_movetable(
		24,
		get_turns_phase2(advanced_turns),
		get_udslice_sorted_coord,
		cube_from_udslice_sorted,
	);

	assert!(udslice_sorted_coord < 24);

	let nc = {
		let mut c = cube.clone();
		for turn in turns.iter() {
			print!("{} ", turn);
			c.apply_turn(*turn);
		}
		println!();
		c
	};

	{
		let car: ArrayCube = nc.clone().into();
		car.print();

		let cperm = nc.get_corner_perm_coord();
		let e8perm = nc.get_edge8_permutation_coord();
		let udslicec = get_udslice_sorted_coord(&nc);

		println!("real: {},{},{}", cperm, e8perm, udslicec);
	}

	println!("PHASE 2");
	println!("{},{},{}", corner_perm, edge8_perm, udslice_sorted_coord);

	let solution = phase2(
		&h2,
		corner_perm,
		&corner_table2,
		edge8_perm,
		&edge8_perm_table,
		udslice_sorted_coord,
		&udslice_movetable2,
		advanced_turns,
	)
	.unwrap();

	for turn in solution {
		turns.push(turn);
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
