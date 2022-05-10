#!/usr/bin/env python3.8



import numpy as np
import random

"""
simulate the monty hall problem, for curiosity
"""

wins = 0;
losses = 0;
for i in range(1000) :

    doors = np.array([1, 2, 3])

    winning_door = random.randint(1, 3)
    chosen_door  = random.randint(1, 3)

    if winning_door == chosen_door :
        reveal_door = doors[doors!=winning_door][random.randint(0,1)]
    else :
        reveal_door = doors[np.logical_and(doors!=winning_door, doors!=chosen_door)][0]

    assert(reveal_door != chosen_door)
    assert(reveal_door != winning_door)

    # assume that we stick with the door
    if winning_door == chosen_door :
        wins = wins + 1;
    else :
        losses = losses + 1

print( "dammit" if wins < losses/1.5 else "ahahah monty is dumb")
