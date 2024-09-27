# rubikscube

A Rubik's Cube library written in Rust.
This library provides basic Rubik's Cube representations and some solving algorithms
to produce an solving sequence.

It's here for interested people who want to find out more about how different solving
algorithms work in detail or for those who want to use basic Rubik's Cube functionality in
their project.

## Simple usage

```rust
use rubikscube::prelude::*;

fn main() {
	let mut cube = CubieCube::new();

	let turns1 = parse_turns("U2 D2 B2 F2 L2 R2").unwrap();
	let turns2 = parse_turns("M2 E2 S2").unwrap();

	cube.apply_turns(turns1);
	cube.apply_turns(turns2);

	assert!(cube.is_solved());
}
```

## Development

This project was created due to a school project, so there is no real plan for
active further development.

But if you'd like to contribute, follow these rules to keep things clean:

- Any new solving algorithm should be its own module in `src/solve/`.
- Any new cube representation should be its own module in `src/cube/`.
- Format code with cargo fmt (and disable it where it makes sense)
- Follow clippy instructions (where it makes sense)

## Solving

This project implemented following algorithm:

- Kociemba's algorithm [https://kociemba.org/cube.htm]
- Thistlewaite's algorithm [https://www.jaapsch.net/puzzles/thistle.htm]

If you want to find out how they work, I'd recommend you go check out the `solve/<algorithm>.rs` files.

**Note for Thistlewaite's algorithm**: That algorithm is super slow and unoptimised, so I generally recommend to
use the Kociemba algorithm.
