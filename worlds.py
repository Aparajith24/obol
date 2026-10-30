"""Every world a task can be set in. A task names its world; the default is the order database."""

import library
import world

WORLDS = {"orders": world.WORLD, "library": library.WORLD}


def world_of(task: dict) -> dict:
    return WORLDS[task.get("world", "orders")]
