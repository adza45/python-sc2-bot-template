#!/usr/bin/env python3
import json
import sys

print("The Python version is {}".format( sys.version_info[:3]))

from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer

from bot import MyBot

def main():
    with open("botinfo.json") as f:
        info = json.load(f)

    race = Race[info["race"]]

    run_game(maps.get("Abyssal Reef LE"), [
        Bot(race, MyBot(use_model=False)),
        Computer(Race.Zerg, Difficulty.Medium)
    ], realtime=False, game_time_limit=(60*45), save_replay_as="test.SC2Replay")

if __name__ == '__main__':
    main()
