"""Implements functions related to playing back the simulation."""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def playback(*, system_events: list[dict]):
    """Plot an animation showing the planar tag game.

    Parameters
    ----------
    system_events: list[dict]
        A list of dictionaries that contain the state history.
    """

    def data_gen():
        for _s in system_events:
            _p = _s["pursuer_state"]
            _t = _s["target_state"]

            ppos = _p.position
            tpos = _t.position

            yield (ppos, tpos)

    def init():
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        return pursuer, target

    def run(data):
        _ppos, _tpos = data
        pursuer.set_data(_ppos)
        target.set_data(_tpos)

        return pursuer, target

    fig, ax = plt.subplots()
    (pursuer,) = ax.plot([], [], "bo")
    (target,) = ax.plot([], [], "ro")

    ani = animation.FuncAnimation(fig, run, frames=data_gen, init_func=init)
    plt.show()
