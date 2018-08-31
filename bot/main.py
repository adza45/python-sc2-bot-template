#!/usr/bin/env python3
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
import keras
import math

#test
HEADLESS = False

os.environ["SC2PATH"] = 'G:/Games/Battlenet/StarCraft II'

class MyBot(sc2.BotAI):
	def __init__(self, use_model=False):
		self.ITERATIONS_PER_MINUTE = 165
		self.MAX_WORKERS = 70
		self.has_scouted = False
		self.do_something_after = 0
		self.train_data = []
		self.siege_tanks = {}
		self.has_found_command_center = False
		self.use_model = use_model
		self.title = 1
		self.siege_tanks = {}

		self.choices = {
						0: self.defend_command_center,
						1: self.attack_known_enemy_unit,
						2: self.attack_known_enemy_structure,
						3: self.do_nothing
						}

		if self.use_model:
			print("USING MODEL!")
			self.model = keras.models.load_model("BasicCNN-30-epochs-0.0001-LR-STAGE1")

	with open(Path(__file__).parent / "../botinfo.json") as f:
		NAME = json.load(f)["name"]

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

	async def get_main_command_center(self):
		if self.units(COMMANDCENTER).ready.exists:
			main_command_center = self.units(COMMANDCENTER).ready.random
			self.main_command_center = main_command_center
			print("Command Center Position {}".format(self.main_command_center))

	def random_location_variance(self, enemy_start_location):
		x = enemy_start_location[0]
		y = enemy_start_location[1]

		# x += ((random.randrange(-20, 20))/100) * enemy_start_location[0]
		# y += ((random.randrange(-20, 20))/100) * enemy_start_location[0]
		x += random.randrange(-5,5)
		y += random.randrange(-5,5)

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
		for marine in self.units(MARINE):
			print("Marine Orders: {}".format(marine.orders))

		game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)

		# draw_dict = {
		#              COMMANDCENTER: [15, (0, 255, 0)],
		#              SUPPLYDEPOT: [2, (20, 235, 0)],
		#              SCV: [1, (55, 200, 0)],
		#              BARRACKS: [3, (55, 200, 0)],
		#              MARINE: [1, (255, 50, 0)],
		#              MEDIVAC: [1, (255, 150, 0)],
		#              SIEGETANK: [1, (255, 100, 0)],
		#              SIEGETANKSIEGED: [1, (255, 100, 0)]
		#             }
		#
		# for unit_type in draw_dict:
		#     for unit in self.units(unit_type).ready:
		#         pos = unit.position
		#         cv2.circle(game_data, (int(pos[0]), int(pos[1])), draw_dict[unit_type][0], draw_dict[unit_type][1], -1)
		#
		# main_base_names = ["nexus", "commandcenter", "hatchery"]
		# for enemy_building in self.known_enemy_structures:
		#     pos = enemy_building.position
		#     if enemy_building.name.lower() not in main_base_names:
		#         cv2.circle(game_data, (int(pos[0]), int(pos[1])), 5, (200, 50, 212), -1)
		# for enemy_building in self.known_enemy_structures:
		#     pos = enemy_building.position
		#     if enemy_building.name.lower() in main_base_names:
		#         cv2.circle(game_data, (int(pos[0]), int(pos[1])), 15, (0, 0, 255), -1)
		#
		# for enemy_unit in self.known_enemy_units:
		#     if not enemy_unit.is_structure:
		#         worker_names = ["probe",
		#                         "scv",
		#                         "drone"]
		#         # if that unit is a PROBE, SCV, or DRONE... it's a worker
		#         pos = enemy_unit.position
		#         if enemy_unit.name.lower() in worker_names:
		#             cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (55, 0, 155), -1)
		#         else:
		#             cv2.circle(game_data, (int(pos[0]), int(pos[1])), 3, (50, 0, 215), -1)

		#SCV
		#CommandCenter

		#
		# for siege_tank in self.units(SIEGETANK):
		# 	if siege_tank.tag not in self.siege_tanks:
		# 		self.siege_tanks[siege_tank.tag] = color
		# 		print("Siege Tank tag {}".format(self.siege_tanks[siege_tank.tag]))
		#
		# remove_keys = []
		# for key in self.siege_tanks.keys():
		# 	foundKey = False
		# 	for siege_tank in self.units(SIEGETANK):
		# 		if key == siege_tank.tag:
		# 			foundKey = True
		#
		# 	if not foundKey:
		# 		print("Deleted siege tank tag {}".format(self.siege_tanks[key]))
		# 		remove_keys.append(key)
		#
		# for remove_key in remove_keys:
		# 	del self.siege_tanks[remove_key]


			 # 4376494082
			 # 4365484034
			 # 4376231940
			 #

		for unit in self.units().ready:
			pos = unit.position

			#print("Unit: {}".format(unit.name))
			curr_name = unit.name.upper()
			#print("CURR NAME: {}".format(curr_name))
			if curr_name == "COMMANDCENTER" or "BARRACKS" or "SUPPLYDEPOT" or "FACTORY" or "STARPORT" or "FACTORYTECHLAB" or "STARPORTREACTOR":
				cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(unit.radius*8), (255, 255, 255), math.ceil(int(unit.radius*0.5)))
			elif curr_name == "SCV":
				cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(unit.radius*8), (245, 245, 245), math.ceil(int(unit.radius*0.5)))
			elif curr_name == "MARINE":
				cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(unit.radius*8), (235, 235, 235), math.ceil(int(unit.radius*0.5)))
			elif curr_name == "MEDIVAC":
				cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(unit.radius*8), (225, 225, 225), math.ceil(int(unit.radius*0.5)))
			elif curr_name == "SIEGETANK":
				cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(unit.radius*8), (215, 215, 215), math.ceil(int(unit.radius*0.5)))
			elif curr_name == "SIEGETANKSIEGED":
				cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(unit.radius*8), (205, 205, 205), math.ceil(int(unit.radius*0.5)))
			else:
				print("UKNOWN UNITS {}".format(unit.name))

		for unit in self.known_enemy_units:
			pos = unit.position
			cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(unit.radius*8), (125, 125, 125), math.ceil(int(unit.radius*0.5)))

		try:
			line_max = 50
			mineral_ratio = self.minerals / 1500
			if mineral_ratio > 1.0:
				mineral_ratio = 1.0

			vespene_ratio = self.vespene / 1500
			if vespene_ratio > 1.0:
				vespene_ratio = 1.0

			population_ratio = self.supply_left / self.supply_cap
			if population_ratio > 1.0:
				population_ratio = 1.0

			plausible_supply = self.supply_cap / 200.0

			worker_weight = len(self.units(SCV)) / (self.supply_cap-self.supply_left)
			if worker_weight > 1.0:
				worker_weight = 1.0

			cv2.line(game_data, (0, 19), (int(line_max*worker_weight), 19), (250, 250, 200), 3)  # worker/supply ratio
			cv2.line(game_data, (0, 15), (int(line_max*plausible_supply), 15), (220, 200, 200), 3)  # plausible supply (supply/200.0)
			cv2.line(game_data, (0, 11), (int(line_max*population_ratio), 11), (150, 150, 150), 3)  # population ratio (supply_left/supply)
			cv2.line(game_data, (0, 7), (int(line_max*vespene_ratio), 7), (210, 200, 0), 3)  # gas / 1500
			cv2.line(game_data, (0, 3), (int(line_max*mineral_ratio), 3), (0, 255, 25), 3)  # minerals minerals/1500
		except Exception as e:
			print(str(e))

		# flip horizontally to make our final fix in visual representation:
		grayed = cv2.cvtColor(game_data, cv2.COLOR_BGR2GRAY)
		self.flipped = cv2.flip(grayed, 0)

		resized = cv2.resize(self.flipped, dsize=None, fx=2, fy=2)

		if not HEADLESS:
			if self.use_model:
				cv2.imshow(str(self.title), resized)
				cv2.waitKey(1)
			else:
				cv2.imshow(str(self.title), resized)
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
					await self.build(SUPPLYDEPOT, near=commandcenters.first.position.towards(self.game_info.map_center, 5))

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
					# print("Main Comamnd center position: {}".format(self.main_command_center.position))
					# print("Main Comamnd center position x: {}".format(self.main_command_center.position.x))
					# print("Main Comamnd center position y: {}".format(self.main_command_center.position.y))
					can_place_factory = False
					can_place_addon = False

					while(can_place_factory == False or can_place_addon == False):
						random_xpos = random.randrange(5, 15)
						random_ypos = random.randrange(-10, 10)

						if self.main_command_center.position.x < 100:
							factory_position = position.Point2(position.Pointlike((self.main_command_center.position.x + random_xpos, self.main_command_center.position.y + random_ypos)))
						else:
							factory_position = position.Point2(position.Pointlike((self.main_command_center.position.x - random_xpos, self.main_command_center.position.y + random_ypos)))

						addon_pos = position.Point2(position.Pointlike((factory_position.x + 5,factory_position.y)))
						can_place_factory = await self.can_place(FACTORY, factory_position)
						can_place_addon = await self.can_place(SUPPLYDEPOT, addon_pos)

					# print("Factory Position {}".format(factory_position))
					# print("Add On Position {}".format(addon_pos))
					# print("Can place Factory {}".format(can_place_factory))
					# print("Can place add on {}".format(can_place_addon))
					await self.build(FACTORY, factory_position)

			for factory in self.units(FACTORY).ready.noqueue:
				#if self.can_afford(FACTORYTECHLAB) and not factory.has_add_on and not self.already_pending(FACTORYTECHLAB):
				#minerals 50
				#vespene 25
				#if self.can_afford(FACTORYTECHLAB):
				if self.minerals >= 50 and self.vespene >= 25:
					if not factory.has_add_on and not self.already_pending(FACTORYTECHLAB):
						await self.do(factory.train(FACTORYTECHLAB))

			if self.units(FACTORY).ready.exists and not self.units(STARPORT):
				if self.can_afford(STARPORT) and not self.already_pending(STARPORT):

					can_place_starport = False
					can_place_addon = False

					while(can_place_starport == False or can_place_addon == False):
						random_xpos = random.randrange(7, 15)
						random_ypos = random.randrange(-10, 10)

						if self.main_command_center.position.x < 100:
							starport_position = position.Point2(position.Pointlike((self.main_command_center.position.x + random_xpos, self.main_command_center.position.y + random_ypos)))
						else:
							starport_position = position.Point2(position.Pointlike((self.main_command_center.position.x - random_xpos, self.main_command_center.position.y + random_ypos)))

						addon_pos = position.Point2(position.Pointlike((starport_position.x + 5,starport_position.y)))
						can_place_starport = await self.can_place(STARPORT, starport_position)
						can_place_addon = await self.can_place(SUPPLYDEPOT, addon_pos)


					await self.build(STARPORT, starport_position)
					#await self.build(STARPORT, near=supply_depot)

			for starport in self.units(STARPORT).ready.noqueue:
				if self.minerals >= 50 and self.vespene >= 25:
					if not starport.has_add_on and not self.already_pending(STARPORTREACTOR):
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

	async def do_nothing(self):
		wait = random.randrange(20, 165)
		self.do_something_after = self.iteration + wait

	async def defend_command_center(self):
		aggressive_units= [MARINE, SIEGETANK, SIEGETANKSIEGED]
		if len(self.known_enemy_units) > 0 and len(self.units(COMMANDCENTER)) > 0:
			target = self.known_enemy_units.closest_to(random.choice(self.units(COMMANDCENTER))).position
			for UNIT in aggressive_units:
				for unit in self.units(UNIT).idle:
					await self.do(unit.attack(target))

			for medivac in self.units(MEDIVAC).idle:
				if len(self.units(MARINE)) > 0:
					await self.do(medivac.move(self.units(MARINE).ready.random))

	async def attack_known_enemy_structure(self):
		aggressive_units= [MARINE, SIEGETANK, SIEGETANKSIEGED]
		if len(self.known_enemy_structures) > 0:
			target = random.choice(self.known_enemy_structures)
			for UNIT in aggressive_units:
				for unit in self.units(UNIT).idle:
					await self.do(unit.attack(target))

			for medivac in self.units(MEDIVAC).idle:
				if len(self.units(MARINE)) > 0:
					await self.do(medivac.move(self.units(MARINE).ready.random))

	async def attack_known_enemy_unit(self):
		aggressive_units= [MARINE, SIEGETANK, SIEGETANKSIEGED]
		if len(self.known_enemy_units) > 0 and len(self.units(COMMANDCENTER)) > 0:
			target = self.known_enemy_units.closest_to(random.choice(self.units(COMMANDCENTER))).position
			for UNIT in aggressive_units:
				for unit in self.units(UNIT).idle:
					await self.do(unit.attack(target))

			for medivac in self.units(MEDIVAC).idle:
				if len(self.units(MARINE)) > 0:
					await self.do(medivac.move(self.units(MARINE).ready.random))

	async def enter_siege_mode(self, siege_tank):
		if self.can_afford(SIEGEMODE_SIEGEMODE):
			print("Can afford siege {}".format(siege_tank.tag))
			await self.do(siege_tank.__call__(SIEGEMODE_SIEGEMODE))

	async def exit_siege_mode(self, siege_tank_sieged):
		if self.can_afford(UNSIEGE_UNSIEGE):
			#print("Should siege {}".format(siege_tank_sieged.tag))
			await self.do(siege_tank_sieged.__call__(UNSIEGE_UNSIEGE))

	async def attack(self):
		# for siege_tank in self.units(SIEGETANK):
		#     shouldSiege = self.siegemode_siege(siege_tank)
		#     #if shouldSiege and self.can_afford(SIEGEMODE_SIEGEMODE):
		#     if shouldSiege:
		#         #print("Should siege {}".format(siege_tank.tag))
		#         if self.can_afford(SIEGEMODE_SIEGEMODE):
		#             #print("Can afford siege {}".format(siege_tank.tag))
		#             await self.do(siege_tank.__call__(SIEGEMODE_SIEGEMODE))
		#
		#         #print("Tag: {}, Orders: {}".format(siege_tank.tag, siege_tank.orders))
		#
		# for siege_tank_sieged in self.units(SIEGETANKSIEGED):
		#     shouldUnsiege = self.siegemode_unsiege(siege_tank_sieged)
		#     if shouldUnsiege and self.can_afford(UNSIEGE_UNSIEGE):
		#         #print("Should siege {}".format(siege_tank_sieged.tag))
		#         await self.do(siege_tank_sieged.__call__(UNSIEGE_UNSIEGE))



		aggressive_units= [MARINE, SIEGETANK, SIEGETANKSIEGED]

		if (len(self.units(aggressive_units[0]).idle) > 0) or (len(self.units(aggressive_units[1]).idle) > 0) or (len(self.units(aggressive_units[2]).idle) > 0):
			target = False
			if self.iteration > self.do_something_after:
				if self.use_model:
					prediction = self.model.predict([self.flipped.reshape([-1, 176, 200, 3])])
					choice = np.argmax(prediction[0])
					#print('prediction: ',choice)

					choice_dict = {0: "No Attack!",
								   1: "Attack close to our nexus!",
								   2: "Attack Enemy Structure!",
								   3: "Attack Enemy Start!"}

					print("Choice #{}:{}".format(choice, choice_dict[choice]))

				else:
					# siege_tanks_length = len(self.units(SIEGETANK))
					# print("siege_tanks_length: {}".format(siege_tanks_length))
					#
					# siege_tanks_sieged_length = len(self.units(SIEGETANKSIEGED))
					#
					# choices_length = len(self.choices)
					# print("choices_length: {}".format(choices_length))
					#
					# total_choices_length = choices_length + siege_tanks_length + siege_tanks_sieged_length
					# print("total_choices_length: {}".format(total_choices_length))

					# index = 4
					# for siege_tank in self.units(SIEGETANK):
					# 	self.choices[index] = siege_tank.tag
					# 	index = index + 1

					total_choices_length = 20
					choice = random.randrange(0, total_choices_length)
				try:
					if choice <= 3:
						await self.choices[choice]()
					elif choice > 3 and choice <= 11:
						if (choice - 4) <= len(self.units(SIEGETANK)):
							await self.enter_siege_mode(self.units(SIEGETANK)[choice - 4])
					elif choice > 11 and choice < 20:
						if (choice - 12) <= len(self.units(SIEGETANKSIEGED)):
							await self.exit_siege_mode(self.units(SIEGETANKSIEGED)[choice - 12])

				except Exception as e:
					print(str(e))


				y = np.zeros(20)
				y[choice] = 1
				print(y)
				self.train_data.append([y,self.flipped])
