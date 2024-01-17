use crate::cube::*;
use strum::*;

const CUBEDATA_LEN: usize = N*N*NUM_SIDES;
type CubeData = [u8; CUBEDATA_LEN];

/// A Rubiks Cube representation, using a single array
/// Fast for turning and low in memory usage
/// Clunky for legality check.
#[derive(Clone, PartialEq, Eq)]
#[derive(Hash)]
#[derive(std::fmt::Debug)]
pub struct ArrayCube {
    pub data: [u8; CUBEDATA_LEN],
}

impl Default for ArrayCube {
    /// Creates a *solved* rubiks cube!
    fn default() -> Self {
	let mut data = [UNKNOWN; N*N*NUM_SIDES];
	for side in 0..NUM_SIDES {
	    for i in 0..(N*N) {
		data[side*(N*N) + i] = side as u8;
	    }
	}

	Self { data }
    }
}


/// Chain 2 transformations (t1 and t2) to one transformation.
/// It returns a new transformation, in which first t1 is applied, then t2.
const fn chain_transform(t1: CubeData, t2: CubeData) -> CubeData {
    let mut out = [0; CUBEDATA_LEN];

    let mut i = 0;
    loop {
	if i == CUBEDATA_LEN { break; }
	out[i] = t2[ t1[i] as usize ];
	i += 1;
    }

    out
}

// ==== TRANSFORMATION MATRICES =====
/*
 The transformation-"matrix".
 Let t be the transformation, s the old state and n the new state:
 n[i] = s[ t[i] ] holds true

 These numbers are carfully selected and shouldn't


 */



/* Unused, but maybe it helps for illustration
const T_BASE: CubeData = [
    0,  1,  2,  3,  4,  5,  6,  7,  8, // up
    9, 10, 11, 12, 13, 14, 15, 16, 17, // down
    18, 19, 20, 21, 22, 23, 24, 25, 26, // back
    27, 28, 29, 30, 31, 32, 33, 34, 35, // front
    36, 37, 38, 39, 40, 41, 42, 43, 44, // left
    45, 46, 47, 48, 49, 50, 51, 52, 53, // right
];
*/

const T_UP: CubeData = [
    6,  3,  0,  7,  4,  1,  8,  5,  2, // up (totally changed)
    9, 10, 11, 12, 13, 14, 15, 16, 17, // down (unchanged)
    36, 37, 38, 21, 22, 23, 24, 25, 26, // back
    45, 46, 47, 30, 31, 32, 33, 34, 35, // front
    27, 28, 29, 39, 40, 41, 42, 43, 44, // left
    18, 19, 20, 48, 49, 50, 51, 52, 53, // right
];

const T_DOWN: CubeData = [
    0,  1,  2,  3,  4,  5,  6,  7,  8, // up
    15, 12,  9, 16, 13, 10, 17, 14, 11, // down (down)
    18, 19, 20, 21, 22, 23, 51, 52, 53, // back
    27, 28, 29, 30, 31, 32, 42, 43, 44, // front
    36, 37, 38, 39, 40, 41, 24, 25, 26, // left
    45, 46, 47, 48, 49, 50, 33, 34, 35, // right
];

const T_FRONT: CubeData = [
    0,  1,  2,  3,  4,  5, 44, 41, 38, // up
    51, 48, 45, 12, 13, 14, 15, 16, 17, // down
    18, 19, 20, 21, 22, 23, 24, 25, 26, // back
    33, 30, 27, 34, 31, 28, 35, 32, 29, // front (done)
    36, 37,  9, 39, 40, 10, 42, 43, 11, // left
    6, 46, 47,  7, 49, 50,  8, 52, 53, // right
];

const T_BACK: CubeData = [
    47, 50, 53,  3,  4,  5,  6,  7,  8, // up
    9, 10, 11, 12, 13, 14, 36, 39, 42, // down
    24, 21, 18, 25, 22, 19, 26, 23, 20, // back (done)
    27, 28, 29, 30, 31, 32, 33, 34, 35, // front
    2, 37, 38,  1, 40, 41,  0, 43, 44, // left
    45, 46, 17, 48, 49, 16, 51, 52, 15, // right
];

const T_LEFT: CubeData = [
    26,  1,  2, 23,  4,  5, 20,  7,  8, // up
    27, 10, 11, 30, 13, 14, 33, 16, 17, // down
    18, 19, 15, 21, 22, 12, 24, 25,  9, // back
    0, 28, 29,  3, 31, 32,  6, 34, 35, // front
    42, 39, 36, 43, 40, 37, 44, 41, 38, // left
    45, 46, 47, 48, 49, 50, 51, 52, 53, // right
];

const T_RIGHT: CubeData = [
    0,  1, 29,  3,  4, 32,  6,  7, 35, // up
    9, 10, 24, 12, 13, 21, 15, 16, 18, // down
    8, 19, 20,  5, 22, 23,  2, 25, 26, // back
    27, 28, 11, 30, 31, 14, 33, 34, 17, // front
    36, 37, 38, 39, 40, 41, 42, 43, 44, // left
    51, 48, 45, 52, 49, 46, 53, 50, 47, // right
];

const T_UP_HALF: CubeData = chain_transform(T_UP, T_UP);
const T_DOWN_HALF: CubeData = chain_transform(T_DOWN, T_DOWN);
const T_BACK_HALF: CubeData = chain_transform(T_BACK, T_BACK);
const T_FRONT_HALF: CubeData = chain_transform(T_FRONT, T_FRONT);
const T_LEFT_HALF: CubeData = chain_transform(T_LEFT, T_LEFT);
const T_RIGHT_HALF: CubeData = chain_transform(T_RIGHT, T_RIGHT);

const T_UP_COUNTER: CubeData = chain_transform(T_UP_HALF, T_UP);
const T_DOWN_COUNTER: CubeData = chain_transform(T_DOWN_HALF, T_DOWN);
const T_BACK_COUNTER: CubeData = chain_transform(T_BACK_HALF, T_BACK);
const T_FRONT_COUNTER: CubeData = chain_transform(T_FRONT_HALF, T_FRONT);
const T_LEFT_COUNTER: CubeData = chain_transform(T_LEFT_HALF, T_LEFT);
const T_RIGHT_COUNTER: CubeData = chain_transform(T_RIGHT_HALF, T_RIGHT);

const T_SLICE_X: CubeData = chain_transform(T_FRONT_COUNTER, T_BACK);
const T_SLICE_X_HALF: CubeData = chain_transform(T_SLICE_X, T_SLICE_X);
const T_SLICE_X_COUNTER: CubeData = chain_transform(T_SLICE_X_HALF, T_SLICE_X);

const T_SLICE_Y: CubeData = chain_transform(T_RIGHT_COUNTER, T_LEFT);
const T_SLICE_Y_HALF: CubeData = chain_transform(T_SLICE_Y, T_SLICE_Y);
const T_SLICE_Y_COUNTER: CubeData = chain_transform(T_SLICE_Y_HALF, T_SLICE_Y);

const T_SLICE_Z: CubeData = chain_transform(T_UP_COUNTER, T_DOWN);
const T_SLICE_Z_HALF: CubeData = chain_transform(T_SLICE_Z, T_SLICE_Z);
const T_SLICE_Z_COUNTER: CubeData = chain_transform(T_SLICE_Z_HALF, T_SLICE_Z);

// The transformation matrices, sorted into an multidimensional list
const TRANSFORM: [[CubeData; NUM_TURN_WISES]; NUM_TURNSIDES] = [
    [T_UP, T_UP_HALF, T_UP_COUNTER],
    [T_DOWN, T_DOWN_HALF, T_DOWN_COUNTER],
    [T_BACK, T_BACK_HALF, T_BACK_COUNTER],
    [T_FRONT, T_FRONT_HALF, T_FRONT_COUNTER],
    [T_LEFT, T_LEFT_HALF, T_LEFT_COUNTER],
    [T_RIGHT, T_RIGHT_HALF, T_RIGHT_COUNTER],

    [T_SLICE_X, T_SLICE_X_HALF, T_SLICE_X_COUNTER],
    [T_SLICE_Y, T_SLICE_Y_HALF, T_SLICE_Y_COUNTER],
    [T_SLICE_Z, T_SLICE_Z_HALF, T_SLICE_Z_COUNTER],
];

// =========

impl RubiksCube for ArrayCube {
    fn apply_turn(&mut self, turn: Turn) {
	// Get the transformation matrix (which is easy because it's carefully sorted)
	let transform = TRANSFORM[ turn.side as usize ][ turn.wise as usize ];

	// Apply the matrix to the current state of cube
	let bef = self.data.clone();
	for i in 0..(N*N*NUM_SIDES) {
	    self.data[i] = bef[ transform[i] as usize ];
	}
    }
}

impl ArrayCube {
    /// Print the cube in the *standard output* with ANSI-colors
    pub fn print(&self) {
	// Print Up-side
	for j in 0..N {
	    print!("       "); // space of 7
	    for i in 0..N {
		print!("{}▀ ", get_color(self.data[ (UP as usize)*(N*N) + i+j*N ]));
	    }
	    print!("\n");
	}

	// Print Left, Front, Right, Back
	const SIDES: [u8; 4] = [LEFT, FRONT, RIGHT, BACK];
	for j in 0..N {
	    for s in SIDES {
		for i in 0..N {
		    print!("{}▄ ", get_color(self.data[ (s as usize)*(N*N) + i+j*N ]));
		}
		print!(" ");
	    }
	    print!("\n");
	}
	print!("\n");

	// Print Down-side
	for j in 0..N {
	    print!("       "); // space of 7
	    for i in 0..N {
		print!("{}▀ ", get_color(self.data[ (DOWN as usize)*(N*N) + i+j*N ]));
	    }
	    print!("\n");
	}
	print!("\x1b[00m");
    }

    /// Apply the given sequence of turns.
    pub fn apply_turns(&mut self, turns: std::vec::Vec<Turn>) {
	for turn in turns {
	    self.apply_turn(turn);
	}
    }

    /// Returns the corner at the position and it's orientation
    /// When you turn the corner piece to it's original place, without turning the front/down, left/right side
    /// an odd number of times, the orientation is as follows:
    /// 0, if it's correctly in it's place
    /// 1, if it's rotated once in counterclockwise
    /// 2, if not 0 or 1, i.e it's rotated clockwise once.
    pub fn get_corner_at_pos( &self, pos: Corner ) -> Option<(Corner, usize)> {
	const fn help(side: Side, x: usize, y: usize) -> usize {
	    side as usize * N*N + x + y*N
	}

	// Get the 3 indices of the corner
	// Not the Corner::URF means, first Up, then Right, then Front,
	// (The order of the characters are relevant!)
	let (i1,i2,i3) = match pos {
	    Corner::URF => (help(UP,2,2), help(RIGHT,0,0), help(FRONT,2,0) ),
	    Corner::UBR => (help(UP,2,0), help(BACK,0,0), help(RIGHT,2,0) ),
	    Corner::DLF => (help(DOWN,0,0), help(LEFT,2,2), help(FRONT,0,2) ),
	    Corner::DFR => (help(DOWN,2,0), help(FRONT,2,2), help(RIGHT,0,2) ),

	    Corner::ULB => (help(UP,0,0), help(LEFT,0,0), help(BACK,2,0) ),
	    Corner::UFL => (help(UP,0,2), help(FRONT,0,0), help(LEFT,2,0) ),
	    Corner::DRB => (help(DOWN,2,2), help(RIGHT,2,2), help(BACK,0,2) ),
	    Corner::DBL => (help(DOWN,0,2), help(BACK,2,2), help(LEFT,0,2) ),
	};

	// Extract the colors
	let cols = [ self.data[i1], self.data[i2], self.data[i3] ];

	let corner = Corner::parse_corner(cols)?;
	let ori = match corner {
	    Corner::URF | Corner::UBR | Corner::ULB | Corner::UFL => {
		if cols[0] == UP { 0 }
		else if cols[1] == UP { 1 }
		else if cols[2] == UP { 2 }
		else { panic!() }
	    },
	    Corner::DLF | Corner::DFR | Corner::DRB | Corner::DBL => {
		if cols[0] == DOWN { 0 }
		else if cols[1] == DOWN { 1 }
		else if cols[2] == DOWN { 2 }
		else { panic!() }
	    }
	};

	Some( (corner, ori as usize) )
    }

    /// Returns the edge at the position and it's orientation
    /// If you would put the edge piece to it's home place without turning the front/back side an
    /// odd number of times, the orientation is:
    /// 0, if it's correctly in it's place
    /// 1, if it's wrong in it's place
    pub fn get_edge_at_pos( &self,  pos: Edge ) -> Option<(Edge, usize)> {
	const fn help(side: Side, x: usize, y: usize) -> usize {
	    side as usize * N*N + x + y*N
	}

	// Get the 2 indices of the edge
	// Note that Edge::UF means: first the Up side, then the Front side
	let (i1,i2) = match pos {
	    Edge::UF => (help(UP,1,2), help(FRONT,1,0)),
	    Edge::UR => (help(UP,2,1), help(RIGHT,1,0)),
	    Edge::UB => (help(UP,1,0), help(BACK,1,0)),
	    Edge::UL => (help(UP,0,1), help(LEFT,1,0)),

	    Edge::DF => (help(DOWN,1,0), help(FRONT,1,2)),
	    Edge::DR => (help(DOWN,2,1), help(RIGHT,1,2)),
	    Edge::DB => (help(DOWN,1,2), help(BACK,1,2)),
	    Edge::DL => (help(DOWN,0,1), help(LEFT,1,2)),

	    Edge::FR => (help(FRONT,2,1), help(RIGHT,0,1)),
	    Edge::BR => (help(BACK,0,1), help(RIGHT,2,1)),
	    Edge::BL => (help(BACK,2,1), help(LEFT,0,1)),
	    Edge::FL => (help(FRONT,0,1), help(LEFT,2,1)),
	};

	// Extract the color
	let cols = [ self.data[i1], self.data[i2] ];

	let edge = Edge::parse_edge(cols)?;
	// Find out the orientation
	let ori = match edge {
	    Edge::UF | Edge::UR | Edge::UB | Edge::UL => cols[0] != UP,
	    Edge::DF | Edge::DR | Edge::DB | Edge::DL => cols[0] != DOWN,
	    Edge::FR | Edge::FL => cols[0] != FRONT,
	    Edge::BR | Edge::BL => cols[0] != BACK,
	};

	Some( (edge, ori as usize) )
    }

    // TODO, this functions may be incomplete! Illegal configurations may be called legal
    /// Check whether the cube configurtion is solvable.
    pub fn is_solvable(&self) -> bool {
	let mut ori: isize = 0;

	// Only an even number of edge can be swapped, if it's odd, the cube is not legal
	for edge in Edge::iter() {
	    let o = match self.get_edge_at_pos(edge) {
		Some((_,o)) => o as isize,
		None => return false,
	    };
	    ori ^= o;
	}
	if (ori & 1) != 0 { return false; }


	// The sum of all the orienation has to be a multiple of 3
	ori = 0;
	for corner in Corner::iter() {
	    let o = match self.get_corner_at_pos(corner) {
		Some((_,o)) => match o {
		    1 => -1,
		    2 => 1,
		    _ => 0,
		},
		None => return false,
	    };

	    ori += o;
	}
	if (ori % 3) != 0 { return false; }

	let mut face_cnt = [0usize; NUM_SIDES];
	for col in self.data { face_cnt[ col as usize] += 1; }
	for cnt in face_cnt {
	    if cnt != N*N { return false; }
	}

	true
    }
}

// =============== //

#[cfg(test)]
mod tests {
    use crate::cube::arraycube::ArrayCube;
    use strum::*;
    use crate::cube::*;

    #[test]
    /// Test for basic turning and their correctness
    fn array_cube_turns1() {
        let mut cube = ArrayCube::default();
	// Little scramble
	cube.apply_turns( parse_turns("L R' U2 F D' R U2 R'") );

	for side in TurnSide::iter() {
	    let turn_n = Turn { side, wise: TurnWise::Clockwise };
	    let turn_c = Turn { side, wise: TurnWise::CounterClockwise };
	    let turn2 = Turn { side, wise: TurnWise::Double };

	    let mut cube_n = cube.clone();
	    cube_n.apply_turn(turn_n);

	    let mut cube_c = cube.clone();
	    cube_c.apply_turn(turn_c);

	    let mut cube2 = cube.clone();
	    cube2.apply_turn(turn2);

	    // Check that every turnwise isn't another one
	    assert_ne!( cube_n, cube2 );
	    assert_ne!( cube2, cube_c );
	    assert_ne!( cube_n, cube_c );

	    // Check that two quarters are equal to one half
	    cube_n.apply_turn(turn_n);
	    assert_eq!(cube_n, cube2);

	    // Check that 3 quarters are equal to one quarter counterclockwise
	    cube_n.apply_turn(turn_n);
	    assert_eq!(cube_n, cube_c);
	}
    }

    #[test]
    /// Test for more basic turning
    fn array_cube_turns2() {
	let mut cube = ArrayCube::default();
	let bef = cube.clone();

	// This sequence should turn back to the solved cube.
	let turns = parse_turns("U2 D2 B2 F2 L2 R2 B2 F2 L2 R2 U2 D2");
	cube.apply_turns(turns);

	assert_eq!(cube, bef);
    }

    #[test]
    /// Test for corner parsing
    fn corner_edge_checking() {
	let mut cube = ArrayCube::default();

	for edge in Edge::iter() {
	    let (e, o) = cube.get_edge_at_pos(edge).unwrap();
	    assert_eq!(e, edge);
	    assert_eq!(o, 0);
	}

	for corner in Corner::iter() {
	    let (c, o) = cube.get_corner_at_pos(corner).unwrap();
	    assert_eq!(c, corner);
	    assert_eq!(o, 0);
	}

	cube.apply_turn( Turn::from("F") );

	let mut cnt = 0;
	for edge in Edge::iter() {
	    let (_e, o) = cube.get_edge_at_pos(edge).unwrap();
	    cnt += o;
	}
	assert_eq!(cnt, 4);
    }
}
