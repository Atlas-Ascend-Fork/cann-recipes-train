"""Progress reporting for expensive per-parameter fingerprints."""

from __future__ import annotations


class FingerprintProgress:
    """Element-weighted tqdm progress; tensor count is shown as context."""

    def __init__(self, desc, *, enabled=True, tqdm_factory=None):
        self.desc = desc
        self.enabled = enabled
        self._tqdm_factory = tqdm_factory
        self._bar = None
        self._total_tensors = 0
        self._done_tensors = 0

    def set_total(self, *, tensors, elements):
        self._total_tensors = tensors
        if not self.enabled:
            return
        if self._tqdm_factory is None:
            from tqdm import tqdm

            self._tqdm_factory = tqdm
        self._bar = self._tqdm_factory(
            total=max(elements, 1),
            desc=self.desc,
            unit="elem",
            unit_scale=True,
            dynamic_ncols=True,
            mininterval=2.0,
        )

    def update(self, *, elements):
        self._done_tensors += 1
        if self._bar is None:
            return
        self._bar.update(elements)
        self._bar.set_postfix_str(
            f"{self._done_tensors}/{self._total_tensors} tensors",
            refresh=False,
        )

    def close(self):
        if self._bar is not None:
            self._bar.close()
