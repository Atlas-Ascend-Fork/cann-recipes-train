"""vime torchtitan recipes, one package per model family.

Each family holds a ``config_registry`` module of no-argument functions returning
``Trainer.Config``, selected with ``--module vime.backends.torchtitan_utils.models.<family>
--config <function>``.
"""
