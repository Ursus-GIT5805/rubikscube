/// Total number of sides
pub const NUM_TURNSIDES: usize = 6 + 3;
/// Total number of ways to adjust your turn
pub const NUM_TURN_WISES: usize = 3;

/// The sides or slices you can turn in a cube
#[derive(Clone, Copy)]
#[derive(PartialEq, Eq, Hash)]
#[derive(Debug)]
#[derive(strum::EnumIter)]
#[repr(u8)]
pub enum TurnSide {
    Up,
    Down,
    Back,
    Front,
    Left,
    Right,
    SliceX, // The slice between front and back. Equivalent to F' * B
    SliceY, // The slice between left and right. Equivalent to R' + L
    SliceZ, // The slice between up and down. Equivalent to U' + D
}

/// You can either turn a side in (Counter-)Clockwise and Half turns
/// This is the enum for that
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
    pub side: TurnSide,
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
	match self.side {
	    TurnSide::Up => write!(f, "U"),
	    TurnSide::Down => write!(f, "D"),
	    TurnSide::Back => write!(f, "B"),
	    TurnSide::Front => write!(f, "F"),
	    TurnSide::Left => write!(f, "L"),
	    TurnSide::Right => write!(f, "R"),
	    TurnSide::SliceX => write!(f, "X"),
	    TurnSide::SliceY => write!(f, "Y"),
	    TurnSide::SliceZ => write!(f, "Z"),
	}?;
	self.wise.fmt(f)
    }
}


impl From<&str> for Turn {
    fn from(item: &str) -> Self {
	if item.len() == 0 || 2 < item.len() {
	    panic!("Turn string has invalid size: \"{}\"", item);
	}
	let bytes = item.as_bytes();

	let side = match bytes[0] as char {
	    'U' => TurnSide::Up,
	    'D' => TurnSide::Down,
	    'B' => TurnSide::Back,
	    'F' => TurnSide::Front,
	    'L' => TurnSide::Left,
	    'R' => TurnSide::Right,
	    'X' => TurnSide::SliceX,
	    'Y' => TurnSide::SliceY,
	    'Z' => TurnSide::SliceZ,
	    _ => panic!("Turn string has an invalid move character!"),
	};

	let wise = if item.len() == 2 {
	    match bytes[1] as char {
		'\'' => TurnWise::CounterClockwise,
		'2'  => TurnWise::Double,
		_ => TurnWise::Clockwise,
	    }
	} else {
	    TurnWise::Clockwise
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
