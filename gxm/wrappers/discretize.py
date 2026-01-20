import jax

from gxm.core import Environment, EnvironmentState, Timestep
from gxm.spaces import Discrete
from gxm.typing import Key, PyTree
from gxm.wrappers.wrapper import Wrapper


class Discretize(Wrapper):
    """
    Wrapper that discretizes a continuous action space.
    Maps a discrete set of actions to the continuous action space of the environment.
    The actions are specified as a list of continuous actions :math:`A`.
    The action space of the wrapped environment is then :math:`\{0, 1, \ldots, |A|-1\}`.

    >>> import gxm
    >>> from gxm.wrappers import Discretize
    >>> env = make("Gymnasium/Pendulum-v1")
    >>> actions = jnp.array([-2.0, 0.0, 2.0])
    >>> env = Discretize(env, actions)

    The actions passed to the ``Discretize`` wrapper need to be of shape :math:`(|A|, D)`,
    where :math:`|A|` is the number of discrete actions and :math:`D` is the dimensionality of the
    continuous action space of the wrapped environment.
    """

    env: Environment
    actions: PyTree

    def __init__(self, env: Environment, actions: PyTree, unwrap: bool = True):
        """
        Args:
            env: The environment to wrap.
            actions: The discrete set of actions to map to.
            unwrap: Whether to unwrap the environment or treat it as part of the base environment.
        """
        super().__init__(env, unwrap=unwrap)
        self.actions = actions
        self.action_space = Discrete(len(actions))

    def init(self, key: Key) -> tuple[EnvironmentState, Timestep]:
        return self.env.init(key)

    def reset(
        self, key: Key, env_state: EnvironmentState
    ) -> tuple[EnvironmentState, Timestep]:
        return self.env.reset(key, env_state)

    def step(
        self,
        key: Key,
        env_state: EnvironmentState,
        action: PyTree,
    ) -> tuple[EnvironmentState, Timestep]:
        action = jax.tree.map(lambda x: x[action], self.actions)
        return self.env.step(key, env_state, action)
