use std::str::FromStr;

use pancurses::*;

use rubikscube::prelude::*;

const OFFSET_X: i32 = 2;
const OFFSET_Y: i32 = 1;

const CUBEDATA_LEN: usize = CUBE_AREA * 6;

/// Draw one side of a cube
fn draw_facelet(win: &Window, x: usize, y: usize, data: &[u8]) {
	let col = if DISPLAY_GRID[y][x] < CUBEDATA_LEN {
		data[DISPLAY_GRID[y][x]] as u32
	} else {
		return;
	};

	let cx = x as i32 * 6 + OFFSET_X;
	let cy = y as i32 * 3 + OFFSET_Y;

	win.attron(COLOR_PAIR(col + 1));
	win.mvprintw(cy, cx, "███");
	win.mvprintw(cy + 1, cx, "▀▀▀");
}

/// Draw the entire cube
fn draw_cube(win: &Window, data: &[u8]) {
	for x in 0..CUBE_DIM * 4 {
		for y in 0..CUBE_DIM * 3 {
			draw_facelet(win, x, y, data);
		}
	}
}

/// Draw cursor at (X/Y). Clear the cursor if clear is true
fn draw_cursor(win: &Window, x: usize, y: usize, clear: bool) {
	let cx = x as i32 * 6 + OFFSET_X;
	let cy = y as i32 * 3 + OFFSET_Y;

	let c = if clear { " " } else { "|" };

	win.attron(COLOR_PAIR(1));
	win.mvprintw(cy, cx - 1, c);
	win.mvprintw(cy, cx + 3, c);
}

fn get_cube(data: &[u8]) -> Result<ArrayCube, ArrayCubeFromStrError> {
	let s: String = data.iter().map(|c| (b'a' + c) as char).collect();
	ArrayCube::from_str(&s)
}

/// Draw the entire screen
fn init(win: &Window, cube: &[u8]) {
	start_color();

	noecho();
	curs_set(0);

	// Set better colors
	init_color(COLOR_WHITE, 1000, 1000, 1000);
	init_color(COLOR_YELLOW, 1000, 1000, 0);
	init_color(COLOR_GREEN, 0, 600, 0);
	init_color(COLOR_BLUE, 67, 67, 933);
	init_color(COLOR_RED, 933, 0, 0);
	init_color(COLOR_MAGENTA, 1000, 612, 12); // it's orange, there is no COLOR_ORANGE

	// Init color pairs
	init_pair(1, COLOR_WHITE, COLOR_BLACK); // white
	init_pair(2, COLOR_YELLOW, COLOR_BLACK); // yellow
	init_pair(3, COLOR_GREEN, COLOR_BLACK); // green
	init_pair(4, COLOR_BLUE, COLOR_BLACK); // blue
	init_pair(5, COLOR_RED, COLOR_BLACK); // red
	init_pair(6, COLOR_MAGENTA, COLOR_BLACK); //orange

	win.clear();
	draw_cube(win, cube);

	win.attron(COLOR_PAIR(1));
	draw_cursor(win, 4, 4, false);

	update_solvability_message(win, cube);

	win.mv(3 * CUBE_DIM as i32 * 3 + 4, 0);

	win.attron(COLOR_PAIR(1));

	win.printw("Move cursor with (i,j,k,l)\n");
	win.printw("Set the color with (w,y|g,b|o,r)\n");
	win.printw("Clear the cube with (shift+)C\n\n");

	win.printw("Press (shift+)Q to quit, if the cube is solvable.");
}

fn get_solvability(data: &[u8]) -> Result<(), String> {
	let arraycube = get_cube(data).map_err(|e| e.to_string())?;
	let cubie: cubiecube::CubieCube = arraycube.try_into().map_err(|e: CubeError| e.to_string())?;
	cubie.check_solvability().map_err(|e| e.to_string())?;
	Ok(())
}

/// Update the message stating the solvability of the cube
fn update_solvability_message(win: &Window, data: &[u8]) {
	win.mv(3 * CUBE_DIM as i32 * 3 + 3, 0);
	win.clrtoeol();

	// Display message depending on the solvabilty of the cube
	match get_solvability(data) {
		Ok(()) => {
			win.attron(COLOR_PAIR(3));
			win.printw("The cube is solvable!");
		}
		Err(e) => {
			win.attron(COLOR_PAIR(5));
			win.printw(e);
		}
	}
}

/// Handle the interactive mode
pub fn interactive_mode() -> String {
	let mut data: Vec<_> = (0..CUBEDATA_LEN as u8).map(|i| i / 9).collect();

	let mut x = 4;
	let mut y = 4;

	let win = initscr();
	init(&win, &data);

	loop {
		if let Some(key) = win.getch() {
			let mut nx = x;
			let mut ny = y;

			if let Input::Character(c) = key {
				match c {
					// Cursor up
					'i' => {
						if y > 0 && DISPLAY_GRID[y - 1][x] < CUBEDATA_LEN {
							ny -= 1;
						}
					}

					// Cursor down
					'k' => {
						if y + 1 < DISPLAY_GRID.len() && DISPLAY_GRID[y + 1][x] < CUBEDATA_LEN {
							ny += 1;
						}
					}

					// Cursor left
					'j' => {
						if x > 0 && DISPLAY_GRID[y][x - 1] < CUBEDATA_LEN {
							nx -= 1;
						}
					}

					// Cursor right
					'l' => {
						if x + 1 < DISPLAY_GRID[y].len() && DISPLAY_GRID[y][x + 1] < CUBEDATA_LEN {
							nx += 1;
						}
					}
					'w' | 'y' | 'g' | 'b' | 'r' | 'o' => {
						let side = match c {
							'w' => Side::Up,
							'y' => Side::Down,
							'g' => Side::Back,
							'b' => Side::Front,
							'r' => Side::Left,
							'o' => Side::Right,
							_ => panic!("Undefined behaviour"),
						} as u8;

						let idx = DISPLAY_GRID[y][x];
						// Check that it isn't the cener piece and else apply it
						if idx % CUBE_AREA != 4 {
							data[idx] = side;
							draw_facelet(&win, x, y, &data);
							update_solvability_message(&win, &data);
						}
					}
					'C' => {
						data = (0..CUBEDATA_LEN)
							.map(|i| i as u8 / CUBE_AREA as u8)
							.collect();
					}
					'Q' => break,
					_ => {}
				}
			}

			if nx != x || ny != y {
				draw_cursor(&win, x, y, true); // clear old cursor
				draw_cursor(&win, nx, ny, false); // draw new cursor

				x = nx;
				y = ny;
			}
		}
	}

	endwin();

	for ele in data.iter_mut() {
		*ele += b'a';
	}

	String::from_utf8(data).unwrap()
}
