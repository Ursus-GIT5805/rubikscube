use rubikscube::prelude::*;

fn main() {
	let mut cube = CubieCube::new();

	let turns1 = parse_turns("U2 D2 B2 F2 L2 R2").unwrap();
	let turns2 = parse_turns("M2 E2 S2").unwrap();

	cube.apply_turns(turns1);
	cube.apply_turns(turns2);

	assert!(cube.is_solved());
}
