import asyncio
from typing import Callable, Dict, List

import gym
import numpy as np
from bomberland.forward_model import ForwardModel
from gym import spaces

# TODO: maybe bit encoding? like
# bits   meaning
# 0-3    type of block (indestructable, destructable wall, empty, enemy, self, pickup boost1, pickup boost2, ...)
# 4+     extra properties
# 4      invulnerable
# 5-7    hp
# 8-10   inventory/bombs
# 11-13  diameter
# ...
# Maybe 16 bit enough?

# TODO: In Efficient Zero converted observation to 8bit
#   File "/home/kftse/EfficientZero/config/atari/env_wrapper.py", line 36, in reset
#     observation = observation.astype(np.uint8)

initial_openai_state = np.zeros((6, 6))

initial_server_state: Dict = {
    "game_id": "dev",
    "agents": {
        "a": {"agent_id": "a", "unit_ids": ["c", "e", "g"]},
        "b": {"agent_id": "b", "unit_ids": ["d", "f", "h"]},
    },
    "unit_state": {
        "c": {
            "coordinates": [0, 1],
            "hp": 3,
            "inventory": {"bombs": 3},
            "blast_diameter": 3,
            "unit_id": "c",
            "agent_id": "a",
            "invulnerability": 0,
        },
        "d": {
            "coordinates": [5, 1],
            "hp": 3,
            "inventory": {"bombs": 3},
            "blast_diameter": 3,
            "unit_id": "d",
            "agent_id": "b",
            "invulnerability": 0,
        },
        "e": {
            "coordinates": [3, 3],
            "hp": 3,
            "inventory": {"bombs": 3},
            "blast_diameter": 3,
            "unit_id": "e",
            "agent_id": "a",
            "invulnerability": 0,
        },
        "f": {
            "coordinates": [2, 3],
            "hp": 3,
            "inventory": {"bombs": 3},
            "blast_diameter": 3,
            "unit_id": "f",
            "agent_id": "b",
            "invulnerability": 0,
        },
        "g": {
            "coordinates": [2, 4],
            "hp": 3,
            "inventory": {"bombs": 3},
            "blast_diameter": 3,
            "unit_id": "g",
            "agent_id": "a",
            "invulnerability": 0,
        },
        "h": {
            "coordinates": [3, 4],
            "hp": 3,
            "inventory": {"bombs": 3},
            "blast_diameter": 3,
            "unit_id": "h",
            "agent_id": "b",
            "invulnerability": 0,
        },
    },
    "entities": [
        {"created": 0, "x": 0, "y": 3, "type": "m"},
        {"created": 0, "x": 5, "y": 3, "type": "m"},
        {"created": 0, "x": 4, "y": 3, "type": "m"},
        {"created": 0, "x": 1, "y": 3, "type": "m"},
        {"created": 0, "x": 3, "y": 5, "type": "m"},
        {"created": 0, "x": 2, "y": 5, "type": "m"},
        {"created": 0, "x": 5, "y": 4, "type": "m"},
        {"created": 0, "x": 0, "y": 4, "type": "m"},
        {"created": 0, "x": 1, "y": 1, "type": "w", "hp": 1},
        {"created": 0, "x": 4, "y": 1, "type": "w", "hp": 1},
        {"created": 0, "x": 3, "y": 0, "type": "w", "hp": 1},
        {"created": 0, "x": 2, "y": 0, "type": "w", "hp": 1},
        {"created": 0, "x": 5, "y": 5, "type": "w", "hp": 1},
        {"created": 0, "x": 0, "y": 5, "type": "w", "hp": 1},
        {"created": 0, "x": 4, "y": 0, "type": "w", "hp": 1},
        {"created": 0, "x": 1, "y": 0, "type": "w", "hp": 1},
        {"created": 0, "x": 5, "y": 0, "type": "w", "hp": 1},
        {"created": 0, "x": 0, "y": 0, "type": "w", "hp": 1},
    ],
    "world": {"width": 6, "height": 6},
    "tick": 0,
    "config": {
        "tick_rate_hz": 10,
        "game_duration_ticks": 300,
        "fire_spawn_interval_ticks": 2,
    },
}


class GymEnv(gym.Env):
    def __init__(
        self,
        fwd_model: ForwardModel,
        channel: int,
        initial_state: Dict,
        send_next_state: Callable[[Dict, List[Dict], int], Dict],
    ):
        self._state = initial_state
        self._initial_state = initial_state
        self._fwd = fwd_model
        self._channel = channel
        self._send = send_next_state
        # [noop, 4 directions, bomb, detonate]
        self.action_space = spaces.Discrete(7)
        # TODO fix observation_space here, should be (map_width)x(map_height)x(# type of block, e.g. wall, bomb, empty ,....)
        high = np.array(
            [
                1,
                2,
                3,
                4,
            ]
        )
        self.observation_space = spaces.Box(-high, high)
        self.loop = asyncio.get_event_loop()

    def reset(self):
        self._state = self._initial_state
        print("Resetting")
        return self._state

    def step(self, actions):
        state = self.loop.run_until_complete(
            self._send(self._state, actions, self._channel)
        )
        self._state = state.get("next_state")
        return [
            state.get("next_state"),
            state.get("is_complete"),
            state.get("tick_result").get("events"),
        ]


class Gym:
    def __init__(self, fwd_model_uri: str):
        self._client_fwd = ForwardModel(fwd_model_uri)
        self._channel_counter = 0
        self._channel_is_busy_status: Dict[int, bool] = {}
        self._channel_buffer: Dict[int, Dict] = {}
        self._client_fwd.set_next_state_callback(self._on_next_game_state)
        self._environments: Dict[str, GymEnv] = {}

    async def connect(self):
        loop = asyncio.get_event_loop()

        client_fwd_connection = await self._client_fwd.connect()

        loop = asyncio.get_event_loop()
        loop.create_task(self._client_fwd._handle_messages(client_fwd_connection))

    async def close(self):
        await self._client_fwd.close()

    async def _on_next_game_state(self, state):
        channel = state.get("sequence_id")
        self._channel_is_busy_status[channel] = False
        self._channel_buffer[channel] = state

    # TODO: should use valid openai_gym_state
    def make(self, name: str, initial_state=initial_openai_state) -> GymEnv:
        if self._environments.get(name) is not None:
            raise Exception(f'environment "{name}" has already been instantiated')
        self._environments[name] = GymEnv(
            self._client_fwd,
            self._channel_counter,
            initial_state,
            self._send_next_state,
        )
        self._channel_counter += 1
        return self._environments[name]

    async def _send_next_state(self, state, actions, channel: int):
        # TODO: convert openai.gym.spaces state to server state
        self._channel_is_busy_status[channel] = True
        await self._client_fwd.send_next_state(channel, state, actions)
        while self._channel_is_busy_status[channel] == True:
            # TODO figure out why packets are not received without some sleep
            await asyncio.sleep(0.0001)
        result = self._channel_buffer[channel]
        # TODO: convert server state to openai.gym.spaces state
        del self._channel_buffer[channel]
        return result
