# rubiks-solver | A rubiks cube solver written in Rust.

---

## Requirements

This projects uses [pancurses](https://github.com/ihalila/pancurses), so you need to install the dependencies of pancurses to make it work.
Under Linux it's `libncurses`.

## Usage

Build it with cargo and for further usage, run it with the flag `-h`.

For short on Linux:

```bash
cargo run -- -h
```

## How it works.

This projects uses the Thistlewaite algorithm to solve the rubik's cube (*see more [here](https://www.jaapsch.net/puzzles/thistle.htm)*).
If you want find out how it works, I'd recommend you go check out the `solve.rs` file.
I've added a lot of commands there, so I think you'll understand.

It is heavily inspired from [this implementation](https://github.com/ldehaudt/Rubik_Solver), written in C++.

## Development

Feel free to open any pull requests.
I will **probably** not maintain this project for a time after the initial commit.

### TODOS

- [ ] Seperate the library for the rubiks cube to it's own repository
