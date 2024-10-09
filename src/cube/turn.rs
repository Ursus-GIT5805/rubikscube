use rand::Rng;
use serde::{Deserialize, Serialize};
use strum::{EnumCount, IntoEnumIterator};

use std::iter::Iterator;
use std::str::FromStr;

use super::Side;

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
	Serialize,
	Deserialize,
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

impl TurnType {
	/// Returns the side it is turning or None if it not exactly one side.
	pub fn get_side(&self) -> Option<Side> {
		let side = match self {
			TurnType::U => Side::Up,
			TurnType::D => Side::Down,
			TurnType::B => Side::Back,
			TurnType::F => Side::Front,
			TurnType::L => Side::Left,
			TurnType::R => Side::Right,
			_ => return None,
		};

		Some(side)
	}
}

/// You can either turn a side in (Counter-)Clockwise and Half turns, that's the wise of a turn.
#[derive(
	Clone,
	Copy,
	PartialEq,
	Eq,
	Hash,
	std::fmt::Debug,
	Serialize,
	Deserialize,
	strum::EnumIter,
	strum::EnumCount,
)]
pub enum TurnWise {
	Clockwise,
	Double,
	CounterClockwise,
}

/// Total number of ways to adjust your turn
pub const NUM_TURNWISES: usize = TurnWise::COUNT;

impl std::fmt::Display for TurnWise {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		match self {
			TurnWise::Clockwise => write!(f, ""),
			TurnWise::CounterClockwise => write!(f, "'"),
			TurnWise::Double => write!(f, "2"),
		}
	}
}

impl TurnWise {
	pub fn inverse(&self) -> Self {
		match self {
			TurnWise::Clockwise => TurnWise::CounterClockwise,
			TurnWise::Double => TurnWise::Double,
			TurnWise::CounterClockwise => TurnWise::Clockwise,
		}
	}
}

// ===== Turn struct =====

/// An entire turn
///
/// name: The annotation for the turn
/// wise: See the definiton of TurnWise
#[derive(Clone, Copy, PartialEq, Eq, Hash, std::fmt::Debug, Serialize, Deserialize)]
pub struct Turn {
	pub typ: TurnType,
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

	pub fn inverse(&self) -> Self {
		Self {
			typ: self.typ,
			wise: self.wise.inverse(),
		}
	}
}

impl std::fmt::Display for Turn {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		self.typ.fmt(f)?;
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

		Ok(Self { typ: side, wise })
	}
}

/// Return two base turns which combines into the given advanced turn
/// or return None if the given turn is not advanced.
pub fn analyze_advanced_turn(turn: Turn) -> Option<(Turn, Turn)> {
	let (s1, s2) = match turn.typ {
		TurnType::M | TurnType::MC => (TurnType::R, TurnType::L),
		TurnType::E | TurnType::EC => (TurnType::D, TurnType::U),
		TurnType::S | TurnType::SC => (TurnType::B, TurnType::F),
		_ => return None,
	};

	let w1 = turn.wise;
	let w2 = match turn.typ {
		TurnType::M | TurnType::E | TurnType::S => w1.inverse(),
		_ => w1,
	};

	let t1 = Turn { typ: s1, wise: w1 };
	let t2 = Turn { typ: s2, wise: w2 };

	Some((t1, t2))
}

/// Parse a sequence of turns from a string
pub fn parse_turns(item: impl Into<String>) -> Result<Vec<Turn>, FromStrError> {
	item.into().split_whitespace().map(Turn::from_str).collect()
}

/// Reverse a sequence of turns
/// ```
/// use rubikscube::prelude::*;
///
/// let turns = random_sequence(20);
/// let inv = reverse_turns(turns.clone());
/// let mut cube = ArrayCube::new();
/// cube.apply_turns(turns);
/// cube.apply_turns(inv);
/// assert!(cube.is_solved());
/// ```
pub fn reverse_turns(turns: Vec<Turn>) -> Vec<Turn> {
	turns.into_iter().rev().map(|t| t.inverse()).collect()
}

/// Return a vector containg every possible turn
pub fn all_turns() -> Vec<Turn> {
	let mut out = vec![];
	for side in TurnType::iter() {
		for wise in TurnWise::iter() {
			out.push(Turn { typ: side, wise });
		}
	}
	out
}

/// Return a random sequence of turns of with length n
pub fn random_sequence(n: usize) -> Vec<Turn> {
	let sides: Vec<_> = TurnType::iter().collect();
	let wises: Vec<_> = TurnWise::iter().collect();
	let mut rng = rand::thread_rng();

	(0..n)
		.map(|_| {
			let side = sides[rng.gen_range(0..NUM_TURNTYPES)];
			let wise = wises[rng.gen_range(0..NUM_TURNWISES)];

			Turn { typ: side, wise }
		})
		.collect()
}
