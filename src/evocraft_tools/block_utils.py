"""
From: Simple_Minecraft_Evolver Tower Evolution:

https://github.com/real-itu/simple_minecraft_evolver
https://arxiv.org/pdf/2012.04751

Setting up Minecraft Server, spawning blocks, etc...

"""


import grpc
import jax
import jax.numpy as jnp
import evocraft_tools.minecraft_pb2_grpc as mcraft_grpc
import numpy as np
import copy

from typing import List, Tuple
from evocraft_tools.minecraft_pb2 import *

block_directions = {"north": NORTH,
                    "west": WEST,
                    "south": SOUTH,
                    "east": EAST,
                    "up": UP,
                    "down": DOWN}

block_direction_codes = lambda direction: block_directions[direction]


def move_coordinate(coord: (int, int, int), side_idx, delta=1):
    """A quick way to increment a coordinate in the desired direction"""
    switcher = [
        lambda c: (c[0], c[1], c[2] - delta),  # Go North
        lambda c: (c[0] - delta, c[1], c[2]),  # Go West
        lambda c: (c[0], c[1], c[2] + delta),  # Go South
        lambda c: (c[0] + delta, c[1], c[2]),  # Go East
        lambda c: (c[0], c[1] + delta, c[2]),  # Go Up
        lambda c: (c[0], c[1] - delta, c[2]),  # Go Down
    ]
    return switcher[side_idx](coord)


class ClientHandler:

    def __init__(self):

        self._blocks = []
        self._channel = grpc.insecure_channel('localhost:5001')
        self._client = mcraft_grpc.MinecraftServiceStub(self._channel)
        self.structure_block = SANDSTONE
        self.block_dict = {
            2: ([RED_SANDSTONE, PUMPKIN], 0),
            3: ([SAND, CACTUS, CACTUS], -1),
            4:CHEST,
            8:GREEN_GLAZED_TERRACOTTA,
            9:BLUE_GLAZED_TERRACOTTA
        }

    def add_block(self, coordinate: (int, int, int), orientation: Orientation, block_type: BlockType):
        assert block_type in BlockType.values(), "Unknown block type"
        assert orientation in Orientation.values(), "Unknown orientation"

        self._blocks.append(Block(
            position=Point(x=coordinate[0], y=coordinate[1], z=coordinate[2]),
            type=block_type,
            orientation=orientation))
        
    def spawn_structure(self, structure, x_offset=0, 
                              y_offset=0, z_offset=0, template=True):
        
        x_grid, z_grid = jnp.meshgrid(jnp.arange(structure.shape[0]),
                                      jnp.arange(structure.shape[1]),
                                      indexing='ij')
        
        for i in range(x_grid.shape[0]):
            for j in range(z_grid.shape[0]):

                x, z = x_grid[i, j], z_grid[i, j] 
                x_spawn, y_spawn, z_spawn = x+x_offset, y_offset, z+z_offset
            
                if template:
                    self.add_block((x_spawn, y_spawn, z_spawn), NORTH, GLASS)
                    y_spawn += 1
   
                block_idx = structure[x,z]


                if block_idx > 0:

                    self.add_block((x_spawn, y_spawn, z_spawn), NORTH, self.structure_block)
                    y_spawn += 1

                    if block_idx > 1: 

                        spawn_block = self.block_dict[int(block_idx)]
                        if isinstance(spawn_block, tuple):

                            y_spawn += spawn_block[1] # where to start spawning
                            blocks = copy.deepcopy(spawn_block[0])

                            while len(blocks) > 0:

                                block = blocks.pop(0)
                                self.add_block((x_spawn, y_spawn, z_spawn), NORTH, block)   
                                y_spawn+=1
                                
                        else:
                            self.add_block((x_spawn, y_spawn, z_spawn), NORTH, spawn_block)   


        self.send_to_server()

    def send_to_server(self):
        response = self._client.spawnBlocks(Blocks(blocks=self._blocks))
        self._blocks = []
        return response

    def fill_cube(self, start_cord: (int, int, int), end_coord: (int, int, int), block_type: BlockType):
        assert block_type in BlockType.values(), "Unknown block type"

        min_x, max_x = (start_cord[0], end_coord[0]) if start_cord[0] < end_coord[0] else (end_coord[0], start_cord[0])
        min_y, max_y = (start_cord[1], end_coord[1]) if start_cord[1] < end_coord[1] else (end_coord[1], start_cord[1])
        min_z, max_z = (start_cord[2], end_coord[2]) if start_cord[2] < end_coord[2] else (end_coord[2], start_cord[2])

        self._client.fillCube(FillCubeRequest(
            cube=Cube(min=Point(x=min_x, y=min_y, z=min_z),
                      max=Point(x=max_x, y=max_y, z=max_z)),
            type=block_type
        ))

    def get_cube_info(self, start_cord: (int, int, int), end_coord: (int, int, int)):
        min_x, max_x = (start_cord[0], end_coord[0]) if start_cord[0] < end_coord[0] else (end_coord[0], start_cord[0])
        min_y, max_y = (start_cord[1], end_coord[1]) if start_cord[1] < end_coord[1] else (end_coord[1], start_cord[1])
        min_z, max_z = (start_cord[2], end_coord[2]) if start_cord[2] < end_coord[2] else (end_coord[2], start_cord[2])

        response = self._client.readCube(Cube(min=Point(x=min_x, y=min_y, z=min_z),
                                              max=Point(x=max_x, y=max_y, z=max_z)))

        return response.blocks

    def spawn_entities(self, entities_and_positions: List[Tuple[int, int, int, int]]):
        spawn_enitites = list(map(
            lambda x: SpawnEntity(type=x[0], spawnPosition=Point(x=x[1], y=x[2], z=x[3])),
            entities_and_positions
        ))
        print(spawn_enitites)
        return self._client.spawnEntities(SpawnEntities(spawnEntities=spawn_enitites))

    def read_entities_in_sphere(self, coord: (int, int, int), radius: float):
        return self._client.readEntitiesInSphere(
            Sphere(center=Point(x=coord[0], y=coord[1], z=coord[2]), radius=radius)
        )

    def read_entities(self, uuids: List[str]):
        parcel = Uuids(uuids=uuids)
        return self._client.readEntities(parcel)