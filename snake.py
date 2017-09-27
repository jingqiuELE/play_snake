# SNAKES GAME
# Use ARROW KEYS to play, SPACE BAR for pausing/resuming and Esc Key for exiting

import curses
import numpy as np
import os
import time
from curses import KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN
from random import randint

KEY_ESC = 27

curses.initscr()

win_height = 20
win_width = 40
win = curses.newwin(win_height, win_width, 0, 0)

win.keypad(1)
curses.noecho()
curses.curs_set(0)
win.border(0)
win.nodelay(1)

key = KEY_RIGHT                                                    # Initializing values
score = 0
key_conflict = {KEY_UP: KEY_DOWN, KEY_DOWN: KEY_UP, KEY_RIGHT: KEY_LEFT, KEY_LEFT: KEY_RIGHT}

snake = [[4,10], [4,9], [4,8]]                                     # Initial snake co-ordinates
food = [10,20]                                                     # First food co-ordinates

win.addch(food[0], food[1], '*')                                   # Prints the food
paused = False

# numpy data collection initialization
m = 5000
count = 0
keymap = {KEY_UP: 0, KEY_DOWN: 1, KEY_RIGHT: 2, KEY_LEFT: 3}
x_data = np.zeros(shape=(win_height, win_width, m))
y_data = np.zeros(shape=(4, m))

def collect_data(win, operation, count):
    for i in range(win_height):
        for j in range(win_width):
            c = win.inch(i, j)
            x_data[i][j][count] = c & curses.A_CHARTEXT
    key_index = keymap[operation]
    y_data[key_index][count] = operation

while True:                                                   # While Esc key is not pressed
    win.border(0)
    win.addstr(0, 2, 'Score : ' + str(score) + ' ')                # Printing 'Score' and
    win.addstr(0, 27, ' SNAKE ')                                   # 'SNAKE' strings
    # Increases the speed of Snake as its length increases
    timeout = 150 - (int(len(snake)/5 + len(snake)/10) % 120)
    win.timeout(timeout)

    if count >= m:
        break

    prevKey = key                                                  # Previous key pressed
    event = win.getch()
    if event == -1:
        key = key
    else:
        key = event
        time.sleep(timeout/1000.0)

    if key == KEY_ESC:
        break

    if key == ord(' '):                                            # If SPACE BAR is pressed, wait for another
        paused = not paused
        key = prevKey

    if paused:
        continue

    if key not in [KEY_LEFT, KEY_RIGHT, KEY_UP, KEY_DOWN]:     # If an invalid key is pressed
        key = prevKey
    else:
        collect_data(win, key, count)
        if prevKey == key_conflict[key]:
            key = prevKey


    # Calculates the new coordinates of the head of the snake. NOTE: len(snake) increases.
    # This is taken care of later at [1].
    dx = 0
    dy = 0
    if key == KEY_DOWN:
        dy = 1
    elif key == KEY_UP:
        dy = -1
    elif key == KEY_RIGHT:
        dx = 1
    elif key == KEY_LEFT:
        dx = -1

    snake.insert(0, [snake[0][0] + dy, snake[0][1] + dx])

    # If snake crosses the boundaries, make it enter from the other side
    if snake[0][0] == 0: snake[0][0] = win_height - 2
    if snake[0][1] == 0: snake[0][1] = win_width - 2
    if snake[0][0] == win_height - 1: snake[0][0] = 1
    if snake[0][1] == win_width - 1: snake[0][1] = 1

    # Exit if snake crosses the boundaries (Uncomment to enable)
    #if snake[0][0] == 0 or snake[0][0] == 19 or snake[0][1] == 0 or snake[0][1] == 59: break

    # If snake runs over itself
    if snake[0] in snake[1:]: break


    if snake[0] == food:                                            # When snake eats the food
        food = []
        score += 1
        while food == []:
            food = [randint(1, win_height - 2), randint(1, win_width - 2)]                 # Calculating next food's coordinates
            if food in snake: food = []
        win.addch(food[0], food[1], '*')
    else:
        last = snake.pop()                                          # [1] If it does not eat the food, length decreases
        win.addch(last[0], last[1], ' ')
    win.addch(snake[0][0], snake[0][1], '#')
    count = count + 1

curses.endwin()

print("\nScore - " + str(score))
print("http://bitemelater.in\n")

if not os.path.exists("data"):
    os.makedirs("data")
np.save('./data/screen.npy', x_data[:, :, 0:count-1])
np.save('./data/operation.npy', y_data[:, 0:count-1])

