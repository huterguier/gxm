import gymnax
from utils import test_environmnet

import gxm

for id in gymnax.registered_envs:
    env = gxm.make("Gymnax/" + id)
    test_environmnet(env)
