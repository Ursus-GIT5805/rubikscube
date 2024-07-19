# rubiks-solver | A Rubik's Cube solver written in Rust.

---

## Requirements

This project uses [pancurses](https://github.com/ihalila/pancurses), so you need to install the dependencies of pancurses to make it work.
Under Linux, it's `libncurses`.

## Usage

Build it with cargo and for further usage, run it with the flag `-h`.

For short on Linux:

```bash
cargo run --release -- -h
```

## Solving

This project implemented following algorithms, which you can change with the `--algorithm` flag:

- Kociembas algorithm (default) [https://kociemba.org/cube.htm]
- Thistlewaites algorithm [https://www.jaapsch.net/puzzles/thistle.htm]

If you want to find out how they work, I'd recommend you go check out the `solve/<algorithm>.rs` files.

If you want to enter the cube, and then get a solving sequence, enter:

```bash
cargo run --release -- -i --solve
```

In the base directory.

## Development

Feel free to open any pull requests.

Any new solving algorithm should be its own file in `src/solve/`.
Any new cube representation should be its own file in `src/cube/`.

## Goals

- [ ] Separate library and cli-tool to their own repositories.
- [ ] Migrate from `pancurses`
