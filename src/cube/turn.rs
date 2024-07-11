use strum::EnumCount;

use std::str::FromStr;

/// Total number of ways to adjust your turn
pub const NUM_TURN_WISES: usize = 3;

/// The sides or slices you can turn in a cube
#[derive(Clone, Copy)]
#[derive(PartialEq, Eq, Hash)]
#[derive(Debug)]
#[derive(strum::EnumIter, strum::EnumCount, strum::EnumString)]
#[repr(u8)]
pub enum Turntype {
    U, // Up
	D, // Down
    B, // Back
	F, // Front
	L, // Left
	R, // Right

	S, // The slice between front and back. Equivalent to F' * B
	M, // The slice between left and right. Equivalent to R' * L
	E, // The slice between up and down. Equivalent to U' * D
}

/// Total number of turntypes
pub const NUM_TURNTYPES: usize = Turntype::COUNT;

impl std::fmt::Display for Turntype {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		write!(f, "{}", self.to_string())
    }
}

/// You can either turn a side in (Counter-)Clockwise and Half turns, that's the wise of a turn.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[derive(strum::EnumIter)]
#[derive(std::fmt::Debug)]
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

/// An entire turn
///
/// side: The side (or slice) to turn a cube
/// wise: See the definiton of TurnWise
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[derive(std::fmt::Debug)]
pub struct Turn {
    pub side: Turntype,
    pub wise: TurnWise,
}

impl Turn {
    /// Turn itself to the turn, which negates itself.
    /// In terms of set theory, convert itself the inverse operation of the current one.
    pub fn invert(&mut self) {
		match self.wise {
			TurnWise::CounterClockwise => self.wise = TurnWise::Clockwise,
			TurnWise::Clockwise => self.wise = TurnWise::CounterClockwise,
			_ => {},
		}
    }
}

impl std::fmt::Display for Turn {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		self.side.fmt(f)?;
		self.wise.fmt(f)
    }
}

impl From<&str> for Turn {
    fn from(item: &str) -> Self {
		let substr = {
			if item.ends_with("'") || item.ends_with("2") {
				&item[0..item.len()-1]
			} else {
				&item[0..item.len()]
			}
		};

		let side = Turntype::from_str( substr ).unwrap();

		let wise = {
			if item.ends_with("'") { TurnWise::CounterClockwise }
			else if item.ends_with("2") { TurnWise::Double }
			else { TurnWise::Clockwise }
		};

		Self { side, wise }
    }
}

impl From<String> for Turn {
    fn from(item: String) -> Self {
		Self::from(item.as_str())
    }
}

pub fn parse_turns<T>(string: T) -> std::vec::Vec<Turn>
where T: Into<String> {
    string.into().split_whitespace().map(Turn::from).collect()
}
