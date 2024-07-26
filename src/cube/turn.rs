use strum::EnumCount;

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

impl FromStr for Turn {
	type Err = ();

	fn from_str(s: &str) -> Result<Self, Self::Err> {
		let substr = {
			if s.ends_with("'") || s.ends_with("2") {
				&s[0..s.len() - 1]
			} else {
				&s[0..s.len()]
			}
		};

		let side = match TurnType::from_str(substr) {
			Ok(res) => res,
			Err(_) => return Err(()),
		};

		let wise = {
			if s.ends_with("'") {
				TurnWise::CounterClockwise
			} else if s.ends_with("2") {
				TurnWise::Double
			} else {
				TurnWise::Clockwise
			}
		};

		Ok(Self { side, wise })
	}
}

pub fn parse_turns<T>(item: T) -> Result<Vec<Turn>, ()>
where
	T: Into<String>,
{
	let out: Result<_, _> = item
		.into()
		.split_whitespace()
		.map(|s| Turn::from_str(s))
		.collect();

	Ok(out?)
}
