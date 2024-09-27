use std::str::FromStr;

use clap::Parser;
use cubiecube::CubieCube;
use strum::{Display, IntoEnumIterator};

mod cube;

#[cfg(feature = "interactive")]
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
	/// Entered sequences or shuffles are ignored
	#[cfg(feature = "interactive")]
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

	/// Output length of sequence (if --solve is used)
	#[arg(short, long, default_value_t = false)]
	length: bool,

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
	#[arg(short, long, default_value_t = String::new())]
	output: String,

	/// Specify whether to use advanced turns for solving
	#[arg(long, default_value_t = false)]
	advanced_turns: bool,
}

fn main() -> std::io::Result<()> {
	#[cfg(debug_assertions)]
	std::env::set_var("RUST_BACKTRACE", "1");

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
		cube = CubieCube::random().into();
	}

	// Parses a cube out of the cube string
	if !args.set.is_empty() {
		cube =
			ArrayCube::from_str(args.set.as_str()).expect("Given cube string couldn't be parsed");
	}

	// Applies turns from args
	cube.apply_turns(parse_turns(args.sequence).expect("Given input sequence could not be parsed"));

	#[cfg(feature = "interactive")]
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
		let cubie: CubieCube = match cube.clone().try_into() {
			Ok(c) => c,
			Err(e) => panic!("The given cube is not solvable: {}", e),
		};
		if let Err(e) = cubie.check_solvability() {
			panic!("The given cube is not solvable: {}", e);
		}

		// Choose algorithm to use
		let seq = match args.algorithm {
			SolveAlgorithm::Thistlewaite => solve::thistlewhaite::solve(cube),
			SolveAlgorithm::Kociemba => solve::kociemba::solve(cube, args.advanced_turns),
		};

		match seq {
			Some(turns) => {
				let len = turns.len();
				for turn in turns {
					write!(out.as_mut(), "{} ", turn)?;
				}
				if args.length {
					writeln!(out.as_mut(), "(len={})", len)?;
				} else {
					writeln!(out.as_mut())?;
				}
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
