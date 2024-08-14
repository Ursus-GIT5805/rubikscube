use rand::Rng;
use strum::{EnumCount, IntoEnumIterator};

use std::iter::Iterator;
use std::str::FromStr;

/// Total number of ways to adjust your turn
pub const NUM_TURNWISES: usize = 3;

/// The sides or slices you can turn in a cube
#[derive(
	Clone,
	Copy,
	PartialEq,
	Eq,
	Hash,
	Debug,
	strum::EnumIter,
	strum::EnumCount,
	strum::EnumString,
	strum::Display,
)]
#[repr(u8)]
pub enum TurnType {
	U, // Up
	D, // Down
	B, // Back
	F, // Front
	L, // Left
	R, // Right

	// Advanced turntypes
	M, // R * L' (middle)
	E, // D * U' (equator)
	S, // B * F' (side)

	MC, // R * L (middle counterdirected)
	EC, // D * U (equator counterdirected)
	SC, // B * F (side counterdirected)
}

/// Total number of turntypes
pub const NUM_TURNTYPES: usize = TurnType::COUNT;

/// You can either turn a side in (Counter-)Clockwise and Half turns, that's the wise of a turn.
#[derive(Clone, Copy, PartialEq, Eq, Hash, strum::EnumIter, std::fmt::Debug)]
pub enum TurnWise {
	Clockwise,
	Double,
	CounterClockwise,
}

impl std::fmt::Display for TurnWise {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		match self {
			TurnWise::Clockwise => write!(f, ""),
			TurnWise::CounterClockwise => write!(f, "'"),
			TurnWise::Double => write!(f, "2"),
		}
	}
}

// ===== Turn struct =====

/// An entire turn
///
/// side: The side/slice or similar to turn
/// wise: See the definiton of TurnWise
#[derive(Clone, Copy, PartialEq, Eq, Hash, std::fmt::Debug)]
pub struct Turn {
	pub side: TurnType,
	pub wise: TurnWise,
}

impl Turn {
	/// Turn itself to the turn, which negates itself.
	/// In terms of set theory, convert itself the inverse operation of the current one.
	pub fn invert(&mut self) {
		match self.wise {
			TurnWise::CounterClockwise => self.wise = TurnWise::Clockwise,
			TurnWise::Clockwise => self.wise = TurnWise::CounterClockwise,
			_ => {}
		}
	}
}

impl std::fmt::Display for Turn {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		self.side.fmt(f)?;
		self.wise.fmt(f)
	}
}

#[derive(thiserror::Error, Debug)]
pub enum FromStrError {
	#[error("Unknown turntype")]
	InvalidTurnType,
}

impl FromStr for Turn {
	type Err = FromStrError;

	fn from_str(s: &str) -> Result<Self, Self::Err> {
		let substr = {
			if s.ends_with('\'') || s.ends_with('2') {
				&s[0..s.len() - 1]
			} else {
				&s[0..s.len()]
			}
		};

		let side = match TurnType::from_str(substr) {
			Ok(res) => res,
			Err(_) => return Err(FromStrError::InvalidTurnType),
		};

		let wise = {
			if s.ends_with('\'') {
				TurnWise::CounterClockwise
			} else if s.ends_with('2') {
				TurnWise::Double
			} else {
				TurnWise::Clockwise
			}
		};

		Ok(Self { side, wise })
	}
}

/// Parse a sequence of turns from a string
pub fn parse_turns<T>(item: T) -> Result<Vec<Turn>, FromStrError>
where
	T: Into<String>,
{
	item.into().split_whitespace().map(Turn::from_str).collect()
}

/// Return a vector containg every possible turn
pub fn all_turns() -> Vec<Turn> {
	let mut out = vec![];
	for side in TurnType::iter() {
		for wise in TurnWise::iter() {
			out.push(Turn { side, wise });
		}
	}
	out
}

/// Return a random sequence of turns of with length n
#[allow(unused)]
pub fn random_sequence(n: usize) -> Vec<Turn> {
	let sides: Vec<_> = TurnType::iter().collect();
	let wises: Vec<_> = TurnWise::iter().collect();
	let mut rng = rand::thread_rng();

	(0..n)
		.map(|_| {
			let side = sides[rng.gen::<usize>() % sides.len()];
			let wise = wises[rng.gen::<usize>() % wises.len()];

			Turn { side, wise }
		})
		.collect()
}
