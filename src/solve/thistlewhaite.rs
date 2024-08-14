use crate::cube::{arraycube::ArrayCube, turn::*, *};
use std::collections::{HashMap, VecDeque};
use strum::*;

/// Simple helper struct to have keep track of the legal moves
#[derive(Default)]
struct TurnSet {
	set: Vec<bool>,
}

impl TurnSet {
	/// Create a full set with all possible turns in the set.
	pub fn new() -> Self {
		let set = vec![true; NUM_TURNTYPES * NUM_TURNWISES];
		Self { set }
	}

	/// Helper function to create the index from a turn
	fn hash(turn: Turn) -> usize {
		(turn.side as usize) * NUM_TURNWISES + turn.wise as usize
	}

	/// Remove the given turn. Does nothing if the turn isn't in the set
	pub fn remove(&mut self, turn: Turn) {
		self.set[Self::hash(turn)] = false;
	}
	/// Remove all turns in the set which is in the given vector
	pub fn vec_remove(&mut self, turns: std::vec::Vec<Turn>) {
		for turn in turns {
			self.remove(turn);
		}
	}
	/// Check whether a turn is in the set
	pub fn has_turn(&self, turn: Turn) -> bool {
		self.set[Self::hash(turn)]
	}
}

/// Use a bfs until it finds a cube, for which hash_fn(cube) == goal holds true.
/// Returns the sequence from which you get to the cube from the cube "initial".
///
/// initial: The inital cube
/// turns: The legal turns you can apply on the cube
/// hasn_fn: The hash_function for the cube
/// goal: The goal hash
fn bfs_solve(
	initial: &mut ArrayCube,
	turns: &TurnSet,
	hash_fn: fn(&ArrayCube) -> u64,
	goal: u64,
) -> Option<Vec<Turn>> {
	let mut queue = VecDeque::<(ArrayCube, Turn)>::new();
	let mut vis = HashMap::<u64, Turn>::new();
	queue.push_front((
		initial.clone(),
		Turn {
			side: TurnType::U,
			wise: TurnWise::Clockwise,
		},
	));

	const MAX_ITERATION: usize = 1_500_000;
	let mut iteration: usize = 0;

	while let Some((state, from)) = queue.pop_back() {
		let hash = hash_fn(&state);

		if vis.contains_key(&hash) {
			continue;
		}
		vis.insert(hash, from);

		iteration += 1;
		// print!("Iteration {}\r", iteration);
		if MAX_ITERATION < iteration {
			return None;
		}

		if hash == goal {
			// println!("Solution iteration {}", iteration);
			let mut out = std::vec::Vec::<Turn>::new();
			let mut cube = state.clone();

			while cube != *initial {
				let hash = hash_fn(&cube);

				if let Some(t) = vis.get(&hash) {
					let mut turn = *t;
					out.push(turn);
					turn.invert();
					cube.apply_turn(turn);
				} else {
					break;
				}
			}

			*initial = state;

			out.reverse();
			return Some(out);
		}

		for side in TurnType::iter() {
			for wise in TurnWise::iter() {
				let turn = Turn { side, wise };
				if !turns.has_turn(turn) {
					continue;
				} // illegal turn

				let mut nextstate = state.clone();
				nextstate.apply_turn(turn);

				queue.push_front((nextstate, turn));
			}
		}
	}

	// println!("Ran out");
	None
}

/// Adjust the legal turns after a phase ends.
///
/// phase: The phase which is going to end.
/// set: The current set of legal turns.
fn end_phase(phase: usize, set: &mut TurnSet) {
	match phase {
		// Phase one is done, that means only F2 and B2 are legal
		// Because of that, set every move that uses F or B quarters to illegal
		1 => set.vec_remove(parse_turns("F F' B B'").unwrap()),
		// Now only R2 and L2 are legal
		2 => set.vec_remove(parse_turns("R R' L L'").unwrap()),
		// Now only U2 and D2 are legal, i.e now only half turns are legal after this
		3 => set.vec_remove(parse_turns("U U' D D'").unwrap()),
		_ => {}
	}
}

fn hash_phase1(cube: &ArrayCube) -> u64 {
	let mut hash: u64 = 0;
	// Put each orientation of all edges in the hash
	// Only front and back quarters change the orientation of an edge.
	// Hence, after this phase, each orientation of an edge has to be 0 (back to normal)
	for edge in Edge::iter() {
		let (_e, o) = cube.get_edge_at_pos(edge).unwrap();
		hash += (o as u64) << (edge as u64);
	}

	hash
}

fn hash_phase2(cube: &ArrayCube) -> u64 {
	let mut hash: u64 = 0;
	// Put each orientation of all corners
	// Each quarter changes the orientation of a corner, except for up and down quarters.
	// Hence, after this phase, each orientation of an corner has to be 0 (back to normal)
	for corner in Corner::iter() {
		let (_c, ori) = cube.get_corner_at_pos(corner).unwrap();
		hash += (ori as u64) << (2 * (corner as u64));
	}

	// Since after this phase, only up and down quarters are legal, each edge in the middle has to be in the middle slice
	// Since all edges are in the correct orientation, the up and down side can only have the colors of the up an down side.
	// You could also say, because only up and down quarters are legal afterwards, the up and down side must only contain up or down colors.
	// Otherwise the cube would be unsolvable!.
	let middle = [Edge::FR, Edge::FL, Edge::BR, Edge::BL];
	for edge in Edge::iter() {
		if middle.contains(&edge) {
			continue;
		}

		hash <<= 1;
		let (e, _) = cube.get_edge_at_pos(edge).unwrap();
		if !middle.contains(&e) {
			hash += 1;
		}
	}

	hash
}

fn hash_phase3(cube: &ArrayCube) -> u64 {
	let mut hash: u64 = 0;

	// Each color of one side must be either the one of their side or the one of the opposite side.
	// The first part of the hash is a bitstring where hash[i] is 1 if the color follows this criteria.
	for i in 0..(9 * NUM_SIDES) {
		hash = (hash << 1) + (cube.data[i] / 18 == (i as u8 / 18)) as u64;
	}

	// Each corner should be on the correct place.
	// We need this to create more distinct hashes for the BFS part
	let mut pos = [Corner::URF; 8];
	for corner in Corner::iter() {
		let (c, _) = cube.get_corner_at_pos(corner).unwrap();
		pos[corner as usize] = c;

		let upper = [Corner::URF, Corner::UBR, Corner::ULB, Corner::UFL];
		let lower = [Corner::DLF, Corner::DFR, Corner::DRB, Corner::DBL];

		let good = (upper.contains(&c) && upper.contains(&corner))
			|| (lower.contains(&c) && lower.contains(&corner));

		hash = (hash << 1) + good as u64;
	}

	hash <<= 1;
	// Only an even number of corners must be swapped, so the last bit is for even or odd number of swaps. 0 is even.
	for i in 0..8 {
		for j in (i + 1)..8 {
			hash ^= (pos[i] as usize > pos[j] as usize) as u64;
		}
	}

	hash
}

fn hash_phase4(cube: &ArrayCube) -> u64 {
	let mut hash: u64 = 0;

	// The entire cube must be solved.
	// So the hash is basically a bitstring where hash[i] is 1 if the color is correct.
	for i in 0..(9 * NUM_SIDES) {
		hash = (hash << 1) + (cube.data[i] == i as u8) as u64;
	}

	hash
}

/// Do optimizations to the solving sequence to shorten in.
fn post_solve_optimization(turns: std::vec::Vec<Turn>) -> std::vec::Vec<Turn> {
	let mut out: std::vec::Vec<Turn> = vec![];

	for turn in turns {
		if let Some(t) = out.last() {
			// If 2 following turns turn the same side, combine it into one single turn
			if t.side == turn.side {
				let new = (t.wise as usize + turn.wise as usize + 2) % (NUM_TURNWISES + 1);
				let idx = out.len() - 1;
				match new {
					0 => {
						let _ = out.pop();
					}
					1 => out[idx].wise = TurnWise::Clockwise,
					2 => out[idx].wise = TurnWise::Double,
					3 => out[idx].wise = TurnWise::CounterClockwise,
					_ => {}
				}
				continue;
			}
		}

		out.push(turn);
	}

	out
}

/// Solve the cube using the thistlewaite algorithm and return the solving-sequence.
///
/// cube: The cube to solve
pub fn solve(cube: ArrayCube) -> Option<Vec<Turn>> {
	let solved = ArrayCube::default();
	let mut solve = cube.clone();
	let mut allowed_moves = TurnSet::new();
	let mut seq = std::vec::Vec::<Turn>::new();

	// Iterate over each phase
	for phase in 1..=4 {
		let hash_fn = match phase {
			1 => hash_phase1,
			2 => hash_phase2,
			3 => hash_phase3,
			4 => hash_phase4,
			_ => break, // this should never happen
		};

		// Set the goal hash (which is just the hash of the solved cube)
		let goal = hash_fn(&solved);

		// Do the BFS
		match bfs_solve(&mut solve, &allowed_moves, hash_fn, goal) {
			Some(turns) => {
				// Push the turns to the output sequence
				for turn in turns {
					seq.push(turn);
				}
			}
			None => return None,
		}

		// Disallow certain turns
		end_phase(phase, &mut allowed_moves);
	}

	let mut test = cube;
	test.apply_turns(seq.clone());
	assert_eq!(test, solved); // superfluous, but to be sure...
	seq = post_solve_optimization(seq); // Do some optimization to the sequence afterwards.

	Some(seq)
}
