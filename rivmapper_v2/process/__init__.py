"""Mask processing: raw annual DSWE counts → annual channel masks.

Modules:

* ``windows``   reach windows on the river grid; reading raw counts.
* ``synthetic`` synthetic rivers with a known channel, for tests and
  validation.
* ``step``      the ``process_masks`` pipeline step.

Planned (see docs/methods/processing.md): ``hmm``, ``channel``,
``topology``, ``calibrate``, ``qa``.
"""
