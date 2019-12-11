
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


from os import listdir
from os.path import dirname

import karabo_data
from xts import TrainSet, DataRoot, DataSource
from typing import Iterable, Generator, Optional


class KaraboDataProposal(DataRoot):
    def __init__(self, proposal):
        self.proposal = proposal
        self.runs = {}
        self.path = None

    def karabo(self, device=None, key=None):
        return KaraboDataSource(self, device, key)

    def by_run(self, run_id: int) -> Iterable[int]:
        return self._open_run(run_id).train_ids

    def _open_run(self, run_id):
        run = karabo_data.open_run(proposal=self.proposal, run=run_id)
        self.runs[run_id] = frozenset(run.train_ids)

        base = dirname(run.files[0].filename)
        self.path = base[:base.rfind('/')]

        return run

    def find_run(self, target: TrainSet):
        # karabo_data requires the explicit creation of run objects. For
        # now, this code does not support TrainSets spanning multiple
        # runs, but this should be possible with only minor performance
        # impact through proper caching.

        # First, we hope to locate the run by checking the first entry
        # against already loaded runs
        for run_id, train_ids in self.runs.items():
            if target.train_ids[0] in train_ids:
                return karabo_data.open_run(proposal=self.proposal, run=run_id)

        # If unsuccessful, load more runs into the cache.

        if self.path is None:
            run = self._open_run(1)

            if target.train_ids[0] in self.runs[1]:
                return run

        for entry in listdir(self.path):
            run_id = int(entry[1:])

            if run_id not in self.runs:
                run = self._open_run(run_id)

                if target.train_ids[0] in self.runs[run_id]:
                    return run

        raise ValueError('unable to locate run ID for TrainSet')


class KaraboDataSource(DataSource):
    def __init__(self, root: KaraboDataProposal, device: Optional[str] = None,
                 key: Optional[str] = None) -> None:
        self.root = root
        self.device = device
        self.key = key

        if self.device is not None and self.key is not None:
            self._extract = lambda x: x[self.device][self.key]
            self._select_args = (self.device, self.key)

        elif self.device is not None:
            self._extract = lambda x: x[self.device]
            self._select_args = (self.device, '*')
        else:
            self._extract = lambda x: x
            self._select_args = ('*', '*')

    def walk_trains(self, target: TrainSet) -> Generator:
        dc = self.root.find_run(target).select(*self._select_args)
        train_it = dc.trains(train_range=karabo_data.by_id[target.train_ids],
                             require_all=True)

        for train_id, data in train_it:
            yield self._extract(data)

    def walk_records(self, target: TrainSet) -> Generator:
        dc = self.root.find_run(target).select(self.devices)
        combined_set = (set(target.train_ids) & set(dc.train_ids))

        for train_id in combined_set:
            yield train_id
