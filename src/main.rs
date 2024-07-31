use std::str::FromStr;

use clap::Parser;
use cubiecube::{CubieCube, CORNER_ORI, CORNER_PERM, EDGE_ORI, EDGE_PERM};
use math::count_permutation_inversions;
use rand::prelude::*;
use strum::{Display, IntoEnumIterator};

mod cube;
mod interactive;
pub mod math;
mod solve;

use cube::{arraycube::ArrayCube, turn::*, *};

#[derive(
	PartialEq, Eq, Default, Debug, Display, Copy, Clone, strum::EnumString, strum::EnumIter,
)]
#[repr(usize)]
#[non_exhaustive]
enum SolveAlgorithm {
	#[default]
	Kociemba,
	Thistlewaite,
}

/// Rubik's Cube solver written in Rust
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
	/// Enter the cube interactively
	#[arg(short, long, default_value_t = false)]
	interactive: bool,

	/// Use a sequence to apply on the solved cube
	#[arg(short, default_value_t = String::new())]
	sequence: String,

	/// Set the cube from a string (the same format as when you output the cube via the "-c"-flag)
	#[arg(long, default_value_t = String::new())]
	set: String,

	/// Solve the cube (the output is a sequence)
	#[arg(long, default_value_t = false)]
	solve: bool,

	/// Output the cube as a string rather than colored
	#[arg(short, long, default_value_t = false)]
	char_print: bool,

	/// Scramble the cube
	#[arg(short, long, default_value_t = false)]
	random: bool,

	/// Specify the algorithm used for solving
	#[arg(long, default_value_t = SolveAlgorithm::default())]
	algorithm: SolveAlgorithm,

	/// Print all possible algorithms and quit
	#[arg(long, default_value_t = false)]
	list_algorithm: bool,

	/// Print the output to a file rather to the stdout
	/// If you want to read the output of the interactive mode, you should use this.
	#[arg(short, long, default_value_t = String::new())]
	output: String,
}

fn main() -> std::io::Result<()> {
	#[cfg(debug_assertions)]
	{
		std::env::set_var("RUST_BACKTRACE", "1");
	}

	let args = Args::parse();
	// Whether to redirect it to the stdout or a file
	let mut out: Box<dyn std::io::Write> = if args.output.is_empty() {
		Box::new(std::io::stdout())
	} else {
		Box::new(std::fs::File::create(args.output)?)
	};
	let mut cube = ArrayCube::default();

	// List the algorithm and exit
	if args.list_algorithm {
		for algo in SolveAlgorithm::iter() {
			writeln!(out, "{}", algo)?;
		}
		return Ok(());
	}

	// Generate a random input cube
	if args.random {
		let mut rng = rand::thread_rng();
		let mut cubie = CubieCube::new();

		// Generate a cubie by setting random coordinates
		cubie.set_edge_orientation(rng.gen::<usize>() % EDGE_ORI);
		cubie.set_corner_orientation(rng.gen::<usize>() % CORNER_ORI);

		let cperm = rng.gen::<usize>() % CORNER_PERM;
		let mut eperm = rng.gen::<usize>() % EDGE_PERM;

		// The number of swaps have to be even
		// Which is equivalent to: The number of inversions has to be even.
		let inv = count_permutation_inversions(cperm);
		let inv2 = count_permutation_inversions(eperm);

		if (inv + inv2) % 2 == 1 {
			// It can be proven that the sum over all factoradic digits
			// are the total number of inversions.
			// Using the factoradic number system, we can simply change
			// the second digit by one, which is determined by the first bit.
			eperm ^= 1;
		}

		cubie.set_corner_permutation(cperm);
		cubie.set_edge_permutation(eperm);

		#[cfg(debug_assertions)]
		assert!(cubie.is_solvable());

		cube = cubie.into();
	}

	// Parses a cube out of the cube string
	if !args.set.is_empty() {
		cube =
			ArrayCube::from_str(args.set.as_str()).expect("Given cube string couldn't be parsed");
	}

	// Applies turns from args
	cube.apply_turns(parse_turns(args.sequence).expect("Given input sequence could not be parsed"));

	// Use the interactive mode
	if args.interactive {
		// Run interactive mode
		let res = interactive::interactive_mode();
		// Parse cube given from the interactive mode
		cube = match ArrayCube::from_str(&res) {
			Ok(res) => res,
			Err(e) => panic!("Given cube has invalid properties: {}", e),
		}
	}

	// Solve the cube and only outputs the sequence
	if args.solve {
		let cubie: CubieCube = cube
			.clone()
			.try_into()
			.expect("The given cube couldn't be converted properly");

		if let Err(e) = cubie.check_validity() {
			panic!("The given cube is not solvable: {}", e);
		}

		// Choose algorithm to use
		let seq = match args.algorithm {
			SolveAlgorithm::Thistlewaite => solve::thistlewhaite::solve(cube),
			SolveAlgorithm::Kociemba => solve::kociemba::solve(cube),
		};

		match seq {
			Some(turns) => {
				for turn in turns {
					write!(out.as_mut(), "{} ", turn)?;
				}
				writeln!(out.as_mut())?;
				return Ok(());
			}
			None => {
				panic!("Could not solve given Rubik's Cube!");
			}
		}
	}

	// Print the resulting cube (either as a string or with colors)
	if args.char_print {
		let s: String = cube.into();
		writeln!(out.as_mut(), "{}", s)?;
	} else {
		cube.print();
	}

	Ok(())
}
