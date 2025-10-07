import gymnasium
from utils import _test_environmnet

import gxm


def test_gymnasium():
    # print("Testing Gymnax Environments")
    # for id in gymnasium.registry.keys():
    id = "LunarLander-v3"
    env = gymnasium.make(id)

    env = gxm.make("Gymnasium/" + id)
    _test_environmnet(env)


if __name__ == "__main__":
    test_gymnasium()
