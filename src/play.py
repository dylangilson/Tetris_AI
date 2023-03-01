"""
* Dylan Gilson
* dylan.gilson@outlook.com
* March 1, 2023
"""

import cv2
from tetris import Tetris


def play():
    env = Tetris()
    env.reset()
    env.render(wait_key=False)

    iteration = 0
    while True:
        k = cv2.waitKey(100)  # 10 milliseconds
        env.render(wait_key=False)

        if k == 27:  # Escape ; exit game
            break

        elif k == 119 or k == 87:  # W ; clockwise rotation
            env.move([0, 0], -90)
            env.render(wait_key=False)

        elif k == 97 or k == 65:  # A ; move left
            env.move([-1, 0], 0)
            env.render(wait_key=False)

        elif k == 115 or k == 83:  # S ; move down
            env.move([0, 1], 0)
            env.render(wait_key=False)

        elif k == 100 or k == 68:  # D ; move right
            env.move([1, 0], 0)
            env.render(wait_key=False)

        elif k == 32:  # space
            _, game_over = env.hard_drop(env.current_position, env.current_rotation, render=False)
            env.render(wait_key=False)

            if game_over:
                break

        if iteration > 3:
            env.fall()

            if env.game_over:
                break

            iteration = 0
        else:
            iteration += 1


if __name__ == '__main__':
    play()
    cv2.destroyAllWindows()
    exit(0)
