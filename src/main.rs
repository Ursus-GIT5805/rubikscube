use std::str::FromStr;

use clap::Parser;
use rand::prelude::*;
use strum::{Display, IntoEnumIterator};

mod cube;
mod interactive;
mod solve;
pub mod math;

use cube::{
    *,
    turn::*,
    arraycube::ArrayCube,
};

#[derive(PartialEq, Eq)]
#[derive(Default, Debug)]
#[derive(Display)]
#[derive(Copy, Clone)]
#[derive(strum::EnumString, strum::EnumIter)]
#[repr(usize)]
enum SolveAlgorithm {
	#[default]
	KOCIEMBA,
	THISTLEWAITE,
}

// Using clap for parsing arguments. For more infos about clap, see official docs.
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
    // Whether to redirect it to the stout or a file
    let mut out: Box< dyn std::io::Write > = if args.output.is_empty() {
		Box::new( std::io::stdout() )
    } else {
		Box::new( std::fs::File::create( args.output )? )
    };
    let mut cube = ArrayCube::default();

	// List the algorithm and exit
	if args.list_algorithm {
		for algo in SolveAlgorithm::iter() {
			writeln!(out, "{}", algo)?;
		}
		std::process::exit(0);
	}

    // Shuffles the cube randomly
    if args.random {
		let mut rng = rand::thread_rng();

		const MOVES: usize = 15;

		let moves: std::vec::Vec<TurnType> = TurnType::iter().collect();
		let wises: std::vec::Vec<TurnWise> = TurnWise::iter().collect();

		for _ in 0..MOVES {
			let idx1: usize = rng.gen::<usize>() % moves.len();
			let idx2: usize = rng.gen::<usize>() % wises.len();

			let turn = Turn {
				side: moves[idx1],
				wise: wises[idx2],
			};

			cube.apply_turn(turn);
		}
    }

    // Parses a cube out of the cube string
    const DATA_LEN: usize = NUM_SIDES*CUBE_DIM*CUBE_DIM;
    match args.set.len() {
		0 => {},
		DATA_LEN => {
			cube = ArrayCube::from_str( args.set.as_str() ).unwrap();
		},
		_ => {
			eprintln!("The size of the cube string is incorrect. Set size should be {}", DATA_LEN);
			std::process::exit(1);
		},
    }

    // Applies turns from args
    cube.apply_turns( parse_turns(args.sequence).unwrap() );

    // Use the interactive mode
    if args.interactive {
		let res = interactive::interactive_mode();
		cube = ArrayCube::from_str(&res).unwrap();
    }

    // Solve the cube and only outputs the sequence
    if args.solve {
		let seq = match args.algorithm {
			SolveAlgorithm::THISTLEWAITE => solve::thistlewhaite::solve(cube),
			SolveAlgorithm::KOCIEMBA => solve::kociemba::solve(cube),
		};

		match seq {
			Some(turns) => {
				for turn in turns {
					write!(out.as_mut(), "{} ", turn)?;
				}
				writeln!(out.as_mut())?;
				std::process::exit(0);
			},
			None => {
				eprintln!("Could not solve given Rubik's Cube!");
				std::process::exit(1);
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
