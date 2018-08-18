import json
from pathlib import Path
import os
import sc2
from sc2 import run_game, maps, Race, Difficulty, position, Result
from sc2.player import Bot, Computer
from sc2.constants import COMMANDCENTER, SCV, SUPPLYDEPOT, REFINERY, BARRACKS, MARINE, FACTORY, SIEGETANK, TECHLAB, STARPORT, BUILD_TECHLAB, STARPORTTECHLAB, MEDIVAC, STARPORTREACTOR, FACTORYTECHLAB, SIEGEMODE_SIEGEMODE, UNSIEGE_UNSIEGE, SIEGETANKSIEGED, REAPER
import random
import cv2
import numpy as np
from sc2.ids.ability_id import AbilityId
from sc2 import unit as unit_module
import time


HEADLESS = True

#os.environ["SC2PATH"] = 'G:/Games/Battlenet/StarCraft II'

class MyBot(sc2.BotAI):
    def __init__(self):
        self.ITERATIONS_PER_MINUTE = 165
        self.MAX_WORKERS = 70
        self.has_scouted = False
        self.do_something_after = 0
        self.train_data = []
        self.siege_tanks = {}
        self.has_found_command_center = False

    with open(Path(__file__).parent / "../botinfo.json") as f:
        NAME = json.load(f)["name"]

    async def on_step(self, iteration):
        if iteration == 0:
            await self.chat_send(f"Name: {self.NAME}")

    def on_end(self, game_result):
        print('--- on_end called ---')
        print(game_result)

        if game_result == Result.Victory:
            np.save("train_data/{}.npy".format(str(int(time.time()))), np.array(self.train_data))

    async def on_step(self, iteration):
        self.iteration = iteration
        await self.distribute_workers()
        await self.build_workers()
        await self.build_supply_depot()
        await self.build_refinery()
        await self.expand()
        await self.offensive_force_buildings()
        await self.build_offensive_force()
        await self.intel()
        await self.attack()

        #if (self.iteration % 200) == 0:
        #    print("Iteration {}".format(self.iteration))

        if self.has_scouted == False:
            await self.scout()
            self.has_scouted = True

        if self.has_found_command_center == False:
            await self.get_main_command_center()
            self.has_found_command_center = True
        #print("TECHLAB: {}".format(MARINE))

    async def get_main_command_center(self):
        if self.units(COMMANDCENTER).ready.exists:
            main_command_center = self.units(COMMANDCENTER).ready.random
            self.main_command_center = main_command_center
            print("Command Center Position {}".format(self.main_command_center))

    def random_location_variance(self, enemy_start_location):
        x = enemy_start_location[0]
        y = enemy_start_location[1]

        x += ((random.randrange(-20, 20))/100) * enemy_start_location[0]
        y += ((random.randrange(-20, 20))/100) * enemy_start_location[0]

        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x > self.game_info.map_size[0]:
            x = self.game_info.map_size[0]
        if y > self.game_info.map_size[1]:
            y = self.game_info.map_size[1]

        go_to = position.Point2(position.Pointlike((x,y)))
        return go_to

    async def scout(self):
        if len(self.units(SCV)) > 0:
            scout = self.units(SCV)[0]
            enemy_location = self.enemy_start_locations[0]
            move_to = self.random_location_variance(enemy_location)
            #print(move_to)
            await self.do(scout.move(move_to))

    async def intel(self):
        game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)

        draw_dict = {
                     COMMANDCENTER: [15, (0, 255, 0)],
                     SUPPLYDEPOT: [2, (20, 235, 0)],
                     SCV: [1, (55, 200, 0)],
                     BARRACKS: [3, (55, 200, 0)],
                     MARINE: [1, (255, 50, 0)],
                     MEDIVAC: [1, (255, 150, 0)],
                     SIEGETANK: [1, (255, 100, 0)],
                     SIEGETANKSIEGED: [1, (255, 100, 0)]
                    }

        for unit_type in draw_dict:
            for unit in self.units(unit_type).ready:
                pos = unit.position
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), draw_dict[unit_type][0], draw_dict[unit_type][1], -1)

        main_base_names = ["nexus", "commandcenter", "hatchery"]
        for enemy_building in self.known_enemy_structures:
            pos = enemy_building.position
            if enemy_building.name.lower() not in main_base_names:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 5, (200, 50, 212), -1)
        for enemy_building in self.known_enemy_structures:
            pos = enemy_building.position
            if enemy_building.name.lower() in main_base_names:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 15, (0, 0, 255), -1)

        for enemy_unit in self.known_enemy_units:
            if not enemy_unit.is_structure:
                worker_names = ["probe",
                                "scv",
                                "drone"]
                # if that unit is a PROBE, SCV, or DRONE... it's a worker
                pos = enemy_unit.position
                if enemy_unit.name.lower() in worker_names:
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (55, 0, 155), -1)
                else:
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), 3, (50, 0, 215), -1)

        line_max = 50
        mineral_ratio = self.minerals / 1500
        if mineral_ratio > 1.0:
            mineral_ratio = 1.0

        vespene_ratio = self.vespene / 1500
        if vespene_ratio > 1.0:
            vespene_ratio = 1.0

        # if self.supply_cap is not 0:
        #     population_ratio = self.supply_left / self.supply_cap
        #     if population_ratio > 1.0:
        #         population_ratio = 1.0

        plausible_supply = self.supply_cap / 200.0

        #military_weight = len(self.units(MARINE)) / (self.supply_cap-self.supply_left)
        #if military_weight > 1.0:
        #    military_weight = 1.0


        #cv2.line(game_data, (0, 19), (int(line_max*military_weight), 19), (250, 250, 200), 3)  # worker/supply ratio
        cv2.line(game_data, (0, 15), (int(line_max*plausible_supply), 15), (220, 200, 200), 3)  # plausible supply (supply/200.0)
        # if population_ratio is not None:
        #     cv2.line(game_data, (0, 11), (int(line_max*population_ratio), 11), (150, 150, 150), 3)  # population ratio (supply_left/supply)
        cv2.line(game_data, (0, 7), (int(line_max*vespene_ratio), 7), (210, 200, 0), 3)  # gas / 1500
        cv2.line(game_data, (0, 3), (int(line_max*mineral_ratio), 3), (0, 255, 25), 3)  # minerals minerals/1500

        # flip horizontally to make our final fix in visual representation:
        self.flipped = cv2.flip(game_data, 0)

        if not HEADLESS:
            resized = cv2.resize(self.flipped, dsize=None, fx=2, fy=2)
            cv2.imshow('Intel', resized)
            cv2.waitKey(1)

    async def build_workers(self):
        if len(self.units(COMMANDCENTER)) * 16 > len(self.units(SCV)):
            if len(self.units(SCV)) < self.MAX_WORKERS:
                for commandcenter in self.units(COMMANDCENTER).ready.noqueue:
                    if self.can_afford(SCV):
                        await self.do(commandcenter.train(SCV))

    async def build_supply_depot(self):
        if self.supply_left < 5 and not self.already_pending(SUPPLYDEPOT):
            commandcenters = self.units(COMMANDCENTER).ready
            if commandcenters.exists:
                if self.can_afford(SUPPLYDEPOT):
                    await self.build(SUPPLYDEPOT, near=commandcenters.first)

    async def build_refinery(self):
        for commandcenter in self.units(COMMANDCENTER).ready:
            vaspenes = self.state.vespene_geyser.closer_than(15.0, commandcenter)
            for vaspene in vaspenes:
                if not self.can_afford(REFINERY):
                    break
                worker = self.select_build_worker(vaspene.position)
                if worker is None:
                    break
                if not self.units(REFINERY).closer_than(1.0, vaspene).exists:
                    await self.do(worker.build(REFINERY, vaspene))

    async def expand(self):
        if self.units(COMMANDCENTER).amount < 4 and self.can_afford(COMMANDCENTER):
            await self.expand_now()

    async def offensive_force_buildings(self):
        if self.units(SUPPLYDEPOT).ready.exists:
            supply_depot = self.units(SUPPLYDEPOT).ready.random

            if len(self.units(BARRACKS)) < ((self.iteration / self.ITERATIONS_PER_MINUTE) / 2):
                if self.can_afford(BARRACKS) and not self.already_pending(BARRACKS):
                    await self.build(BARRACKS, near=supply_depot)

            if self.units(BARRACKS).ready.exists and not self.units(FACTORY):
                if self.can_afford(FACTORY) and not self.already_pending(FACTORY):
                    print("Main Comamnd center position: {}".format(self.main_command_center.position))
                    print("Main Comamnd center position x: {}".format(self.main_command_center.position.x))
                    print("Main Comamnd center position y: {}".format(self.main_command_center.position.y))
                    can_place_factory = False
                    can_place_addon = False

                    while(can_place_factory == False or can_place_addon == False):
                        random_xpos = random.randrange(5, 15)
                        random_ypos = random.randrange(-10, 10)

                        if self.main_command_center.position.x < 100:
                            factory_position = position.Point2(position.Pointlike((self.main_command_center.position.x + random_xpos, self.main_command_center.position.y + random_ypos)))
                        else:
                            factory_position = position.Point2(position.Pointlike((self.main_command_center.position.x - random_xpos, self.main_command_center.position.y + random_ypos)))

                        addon_pos = position.Point2(position.Pointlike((factory_position.x + 2,factory_position.y)))
                        can_place_factory = await self.can_place(FACTORY, factory_position)
                        can_place_addon = await self.can_place(SUPPLYDEPOT, addon_pos)

                    print("Factory Position {}".format(factory_position))
                    print("Add On Position {}".format(addon_pos))
                    print("Can place Factory {}".format(can_place_factory))
                    print("Can place add on {}".format(can_place_addon))
                    await self.build(FACTORY, factory_position)

            for factory in self.units(FACTORY).ready.noqueue:
                #if self.can_afford(FACTORYTECHLAB) and not factory.has_add_on and not self.already_pending(FACTORYTECHLAB):
                if self.can_afford(FACTORYTECHLAB) and not factory.has_add_on and not self.already_pending(FACTORYTECHLAB):
                    await self.do(factory.train(FACTORYTECHLAB))

            if self.units(FACTORY).ready.exists and not self.units(STARPORT):
                if self.can_afford(STARPORT) and not self.already_pending(STARPORT):
                    print("Main Comamnd center position: {}".format(self.main_command_center.position))
                    print("Main Comamnd center position x: {}".format(self.main_command_center.position.x))
                    print("Main Comamnd center position y: {}".format(self.main_command_center.position.y))
                    can_place_starport = False
                    can_place_addon = False

                    while(can_place_starport == False or can_place_addon == False):
                        random_xpos = random.randrange(7, 17)
                        random_ypos = random.randrange(-10, 10)

                        if self.main_command_center.position.x < 100:
                            starport_position = position.Point2(position.Pointlike((self.main_command_center.position.x + random_xpos, self.main_command_center.position.y + random_ypos)))
                        else:
                            starport_position = position.Point2(position.Pointlike((self.main_command_center.position.x - random_xpos, self.main_command_center.position.y + random_ypos)))

                        addon_pos = position.Point2(position.Pointlike((starport_position.x + 2,starport_position.y)))
                        can_place_starport = await self.can_place(STARPORT, starport_position)
                        can_place_addon = await self.can_place(SUPPLYDEPOT, addon_pos)

                    print("Starport Position {}".format(starport_position))
                    print("Add On Position Starport {}".format(addon_pos))
                    print("Can place Starport {}".format(can_place_starport))
                    print("Can place add on Starport {}".format(can_place_addon))
                    await self.build(STARPORT, starport_position)
                    #await self.build(STARPORT, near=supply_depot)

            for starport in self.units(STARPORT).ready.noqueue:
                if self.can_afford(STARPORTREACTOR) and not starport.has_add_on:
                    await self.do(starport.train(STARPORTREACTOR))


    async def build_offensive_force(self):
        for barrack in self.units(BARRACKS).ready.noqueue:
            if self.can_afford(MARINE) and self.supply_left > 0:
                await self.do(barrack.train(MARINE))

        for starport in self.units(STARPORT).ready.noqueue:
            if self.can_afford(MEDIVAC) and self.supply_left > 0:
                await self.do(starport.train(MEDIVAC))

        for factory in self.units(FACTORY).ready.noqueue:
            if self.can_afford(SIEGETANK) and factory.has_add_on and self.supply_left > 0:
                await self.do(factory.train(SIEGETANK))

    def find_target(self, state):
        if len(self.known_enemy_units) > 0:
            return random.choice(self.known_enemy_units)
        elif len(self.known_enemy_structures) > 0:
            return random.choice(self.known_enemy_structures)
        else:
            return self.enemy_start_locations[0]

    def siegemode_siege(self, siege_tank):
        for enemy in self.known_enemy_units:
            distance = siege_tank.distance_to(enemy)
            if distance <= 14:
                return True

        return False
        #await self.do(siege_tank.__call__(SIEGEMODE_SIEGEMODE))

    def siegemode_unsiege(self, siege_tank_sieged):
        for enemy in self.known_enemy_units:
            distance = siege_tank_sieged.distance_to(enemy)
            if distance <= 14:
                return False

        return True

    async def attack(self):
        for siege_tank in self.units(SIEGETANK):
            shouldSiege = self.siegemode_siege(siege_tank)
            #if shouldSiege and self.can_afford(SIEGEMODE_SIEGEMODE):
            if shouldSiege:
                #print("Should siege {}".format(siege_tank.tag))
                if self.can_afford(SIEGEMODE_SIEGEMODE):
                    #print("Can afford siege {}".format(siege_tank.tag))
                    await self.do(siege_tank.__call__(SIEGEMODE_SIEGEMODE))

                #print("Tag: {}, Orders: {}".format(siege_tank.tag, siege_tank.orders))

        for siege_tank_sieged in self.units(SIEGETANKSIEGED):
            shouldUnsiege = self.siegemode_unsiege(siege_tank_sieged)
            if shouldUnsiege and self.can_afford(UNSIEGE_UNSIEGE):
                #print("Should siege {}".format(siege_tank_sieged.tag))
                await self.do(siege_tank_sieged.__call__(UNSIEGE_UNSIEGE))

        aggressive_units= [MARINE, MEDIVAC, SIEGETANK]

        if (len(self.units(aggressive_units[0]).idle) > 0) or (len(self.units(aggressive_units[1]).idle) > 0) or (len(self.units(aggressive_units[2]).idle) > 0):
            choice = random.randrange(0, 4)
            target = False
            if self.iteration > self.do_something_after:
                if choice == 0:
                    # no attack
                    wait = random.randrange(20, 165)
                    self.do_something_after = self.iteration + wait

                elif choice == 1:
                    #attack_unit_closest_nexus
                    if len(self.known_enemy_units) > 0 and len(self.units(COMMANDCENTER)) > 0:
                        target = self.known_enemy_units.closest_to(random.choice(self.units(COMMANDCENTER)))

                elif choice == 2:
                    #attack enemy structures
                    if len(self.known_enemy_structures) > 0:
                        target = random.choice(self.known_enemy_structures)

                elif choice == 3:
                    #attack_enemy_start
                    target = self.enemy_start_locations[0]

                if target:
                    for UNIT in aggressive_units:
                        for unit in self.units(UNIT).idle:
                            await self.do(unit.attack(target))

            y = np.zeros(4)
            y[choice] = 1
            #print(y)
            self.train_data.append([y,self.flipped])
