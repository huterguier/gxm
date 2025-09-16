import envpool
from utils import _test_environmnet

import gxm


def test_gymnax():
    for id in envpool.list_all_envs()[:10]:
        env = gxm.make("Envpool/" + id)
        _test_environmnet(env)
