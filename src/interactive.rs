use pancurses::*;

use crate::cube::{
    *,
    turn::*,
    arraycube::ArrayCube,
};

const SIDE_WIDTH: i32 = 3*6;
const SIDE_HEIGHT: i32 = 3*3;

fn draw_side(win: &Window, y: i32, x: i32, slice: &[u8]) {
    win.mv(y,x);

    for j in 0..3usize {
	win.mv(y+3*(j as i32),x);
	for i in 0..3 {
	    win.attron( COLOR_PAIR(slice[i+j*3] as u32 +1) );
	    win.printw("███");
	    win.attron( COLOR_PAIR(1) );
	    win.printw("   ");
	}
	win.mv(y+3*(j as i32)+1,x);
	for i in 0..3 {
	    win.attron( COLOR_PAIR(slice[i+j*3] as u32 +1) );
	    win.printw("▀▀▀");
	    win.attron( COLOR_PAIR(1) );
	    win.printw("   ");
	}
    }
}

fn draw_cube(win: &Window, cube: &ArrayCube) {
    let offy = 2;
    let offx = offy*2;

    { // Up
	let slice = &cube.data[0..9];
	draw_side(win, offy, offx+SIDE_WIDTH, slice);
    }
    { // Down
	let slice = &cube.data[9..18];
	draw_side(win, offy+2*SIDE_HEIGHT, offx+SIDE_WIDTH, slice);
    }
    { // Back
	let slice = &cube.data[18..27];
	draw_side(win, offy+SIDE_HEIGHT, offx+3*SIDE_WIDTH, slice);
    }
    { // Front
	let slice = &cube.data[27..36];
	draw_side(win, offy+SIDE_HEIGHT, offx+SIDE_WIDTH, slice);
    }
    { // Left
	let slice = &cube.data[36..45];
	draw_side(win, offy+SIDE_HEIGHT, offx, slice);
    }
    { // Right
	let slice = &cube.data[45..54];
	draw_side(win, offy+SIDE_HEIGHT, offx+2*SIDE_WIDTH, slice);
    }
}

/// Handle the interactive mode
pub fn interactive_mode(cube: &mut ArrayCube) {
    /*
     For real, this code is ugly,
     Be warned that you don't see beautiful code here.
    */

    let mut idx = 0usize;

    let win = initscr();

    // noecho();
    start_color();

    win.printw("Press any button.");
    win.refresh();

    loop {
	if let Some(key) = win.getch() {
	    match key {
		Input::Character(c) => match c {
		    'U' | 'D' | 'F' | 'B' | 'L' | 'R' | 'X' | 'Y' | 'Z' => {
			cube.apply_turn( Turn::from(String::from(c)) );
		    },

		    // Cursor up
		    'i' => if (idx % 9) / 3 == 0 {
			match idx as u8 / 9 {
			    FRONT => idx = (idx+54-3) % 9 + (UP as usize)*9,
			    DOWN => idx = (idx+54-3) % 9 + (FRONT as usize)*9,
			    _ => {},
			}
		    } else {
			idx -= 3;
		    },

		    // Cursor down
		    'k' => if (idx % 9) / 3 == 2 {
			match idx as u8 / 9 {
			    FRONT => idx = (idx+3) % 9 + (DOWN as usize)*9,
			    UP => idx = (idx+3) % 9 + (FRONT as usize)*9,
			    _ => {},
			}
		    } else {
			idx += 3;
		    },

		    // Cursor left
		    'j' => if idx % 3 == 0 {
			match idx as u8 / 9 {
			    FRONT => idx = (idx+2) % 9 + (LEFT as usize)*9,
			    RIGHT => idx = (idx+2) % 9 + (FRONT as usize)*9,
			    BACK => idx = (idx+2) % 9 + (RIGHT as usize)*9,
			    _ => {},
			}
		    } else {
			idx -= 1;
		    },


		    // Cursor right
		    'l' => if idx % 3 == 2 {
			match idx as u8 / 9 {
			    LEFT => idx = (idx-2) % 9 + (FRONT as usize)*9,
			    FRONT => idx = (idx-2) % 9 + (RIGHT as usize)*9,
			    RIGHT => idx = (idx-2) % 9 + (BACK as usize)*9,
			    _ => {},
			}
		    } else {
			idx += 1;
		    },


		    'w' => cube.data[idx] = UP,
		    'y' => cube.data[idx] = DOWN,
		    'g' => cube.data[idx] = BACK,
		    'b' => cube.data[idx] = FRONT,
		    'r' => cube.data[idx] = LEFT,
		    'o' => cube.data[idx] = RIGHT,

		    'Q' => break,
		    _ => {},
		},
		_ => continue,
	    }
	} else {
	    continue;
	}

	idx %= 54;

	init_pair(1, 15, COLOR_BLACK);
	init_pair(2, 11, COLOR_BLACK);
	init_pair(3, COLOR_GREEN, COLOR_BLACK);
	init_pair(4, COLOR_BLUE, COLOR_BLACK);
	init_pair(5, COLOR_RED, COLOR_BLACK);
	init_pair(6, 3, COLOR_BLACK);

	win.clear();
	draw_cube(&win, cube);

	let offy = 2;
	let offx = offy*2;

	let (mut y, mut x) = match (idx / 9) as u8 {
	    UP => (0,SIDE_WIDTH),
	    DOWN => (SIDE_HEIGHT*2,SIDE_WIDTH),
	    BACK => (SIDE_HEIGHT,SIDE_WIDTH*3),
	    FRONT => (SIDE_HEIGHT,SIDE_WIDTH),
	    LEFT => (SIDE_HEIGHT,0),
	    RIGHT => (SIDE_HEIGHT,SIDE_WIDTH*2),
	    _ => (0,0),
	};
	x += offx + 6 * (idx as i32 % 3);
	y += offy + 3 * ((idx as i32 / 3) % 3);

	win.mv(y, x-1);
	win.printw("│");
	win.mv(y+1, x-1);
	win.printw("│");

	win.mv(y, x+3);
	win.printw("│");
	win.mv(y+1, x+3);
	win.printw("│");

	win.mv(SIDE_HEIGHT*3+3, 0);
	win.printw("Press any of these keys to use the move (shift+)(U,D,B,F,L,R)\n");
	win.printw("Move cursor with (i,j,k,l)\n");
	win.printw("Set the color with (w,y|g,b|o,r)\n\n");
	win.printw(format!("Index {}\n", idx));

	if cube.is_solvable() {
	    win.printw("This cube is solvable.\n");
	} else {
	    win.printw("This cube is UNSOLBABLE!!!\n");
	}

	win.printw("Press (shift+)Q to quit.");

	win.refresh();
    }

    endwin();
}
