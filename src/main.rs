use clap::Parser;
use rand::prelude::*;
use strum::IntoEnumIterator;

mod cube;
mod interactive;
mod solve;

use cube::{
    *,
    turn::*,
    arraycube::ArrayCube,
};

// Using clap for parsing arguments. For more infos, see official docs.
/// Rubiks cube solver written in Rust
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Play and test the cube interactively
    #[arg(short, long, default_value_t = false)]
    interactive: bool,

    /// Use this sequence the to turn the cube
    #[arg(short, default_value_t = String::new())]
    sequence: String,

    /// Set the cube from a string (the same format as when you output the cube via the "-c"-flag)
    #[arg(long, default_value_t = String::new())]
    set: String,

    /// Solve the cube (the program outputs a sequence)
    #[arg(long, default_value_t = false)]
    solve: bool,

    /// Outputs the cube as a string rather than colored
    #[arg(short, long, default_value_t = false)]
    char_print: bool,

    /// Scrambles the cube randomly (but with a legal state)
    #[arg(short, long, default_value_t = false)]
    random: bool,

    /// Prints the output to a file rather to the stdout
    /// You should write to a temporary file to read the output in the interactive mode.
    #[arg(short, long, default_value_t = String::new())]
    output: String,
}

/// Removes all combined turns.
/// Core turns are all the U,D,L,R,F,B clockwise, double or counterclockwise
/// Turns like X,Y,Z are combinations out of the above ones and are so called combined turns.
fn convert_combined_turns( turns: std::vec::Vec<Turn> )-> std::vec::Vec<Turn> {
    let mut out = vec![];

    for turn in turns {
	// Only handles slice turns, since they are the only one supported now.
	let mut ts = match turn.side {
	    TurnSide::SliceX => vec![Turn::from("F'"), Turn::from("B")],
	    TurnSide::SliceY => vec![Turn::from("R'"), Turn::from("L")],
	    TurnSide::SliceZ => vec![Turn::from("U'"), Turn::from("D")],
	    _ => vec![],
	};

	if !ts.is_empty() {
	    match turn.wise {
		TurnWise::Clockwise => {},
		TurnWise::Double => {
		    ts[0].wise = TurnWise::Double;
		    ts[1].wise = TurnWise::Double;
		},
		TurnWise::CounterClockwise => {
		    ts[0].wise = TurnWise::Clockwise;
		    ts[1].wise = TurnWise::CounterClockwise;
		}
	    }

	    for t in ts { out.push(t); }
	} else {
	    out.push(turn);
	}
    }

    out
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    // Wheter to redirect it to the stout or a file
    let mut out: Box< dyn std::io::Write > = if args.output.is_empty() {
	Box::new( std::io::stdout() )
    } else {
	Box::new( std::fs::File::create( args.output )? )
    };
    let mut cube = ArrayCube::default();

    // Shuffles the cube randomly
    if args.random {
	let mut rng = rand::thread_rng();

	const MOVES: usize = 10;

	for _ in 0..MOVES {
	    let moves: std::vec::Vec<TurnSide> = TurnSide::iter().collect();
	    let wises: std::vec::Vec<TurnWise> = TurnWise::iter().collect();

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
    const DATA_LEN: usize = 54;
    match args.set.len() {
	0 => {},
	DATA_LEN => {
	    let bytes = args.set.as_bytes();
	    for (i, byte) in bytes.iter().enumerate().take(DATA_LEN) {
		let v = byte - b'a';
		cube.data[i] = v;
	    }
	},
	_ => {
	    eprintln!("The size of the cube string is incorrect. Set size should be {}", DATA_LEN);
	    std::process::exit(1);
	},
    }

    // Applies turns from args
    cube.apply_turns(parse_turns(args.sequence));

    // Use the interactive mode
    if args.interactive {
	interactive::interactive_mode(&mut cube);
    }

    // Solve the cube and only outputs the sequence
    if args.solve {
	let turns =  convert_combined_turns( solve::solve(cube) );

	for turn in turns {
	    write!(out.as_mut(), "{} ", turn)?;
	}
	writeln!(out.as_mut())?;
	std::process::exit(0);
    }

    // Print the resulting cube (either as a string or with colors)
    if args.char_print {
	for s in cube.data {
	    let c = (b'a' + s) as char;
	    write!(out.as_mut(), "{}", c)?;
	}
	writeln!(out.as_mut())?;
    } else {
	cube.print();
    }

    Ok(())
}
