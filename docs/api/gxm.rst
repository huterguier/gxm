API Reference of ``gxm``
=========================

Core API
--------

.. autosummary::
   :toctree: _autosummary

   gxm.Timestep
   gxm.Transition
   gxm.Trajectory
   gxm.EnvironmentState
   gxm.Environment

Wrappers
--------

.. autosummary::
   :toctree: _autosummary

   gxm.wrappers.RecordEpisodeStatistics
   gxm.wrappers.FlattenObservation
   gxm.wrappers.StackObservations
   gxm.wrappers.Discretize
   gxm.wrappers.IgnoreTruncation

Spaces
------

.. autosummary::
   :toctree: _autosummary

   gxm.spaces.Discrete
   gxm.spaces.Box
   gxm.spaces.Tree
