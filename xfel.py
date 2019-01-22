
from typing import Optional

import h5py
import os.path
import re

import numpy

from xts import HdfData, IndexedRoot


class KaraboData(HdfData):
    def generate_index(self, raw_file_id, h5f=None, timing_train_ids=None):
        pass


class EuXfelProposal(IndexedRoot):
    def __init__(self, instrument: str, cycle: int, proposal: int,
                 prefix: Optional[str] -> None) -> None:
        self.instrument = instrument
        self.cycle = cycle
        self.proposal = proposal

        if prefix is None:
            prefix = f'exfel_{proposal}'

        self.data_sources = []

        super().__init__(prefix)

    @classmethod
    def auto_detect(cls, path: str = None) -> Optional['EuXfelProposal']:
        if path is None:
            path = os.path.realpath(os.getcwd())

        m = re.match('/gpfs/exfel/exp/(\w{3,6})/(\d{6,6})/p(\d{6,6})', path)

        if m is not None:
            instrument, cycle_str, proposal_str = m.groups()
            return cls(instrument, int(cycle_str), int(proposal_str))

    def get_run_from_path(self, path):
        raise NotImplementedError('get_run_from_path')

    def get_gpfs_root(self) -> str:
        return f'/gpfs/exfel/exp/{self.instrument}/{self.cycle}' \
               f'/p{self.proposal:06d}/raw'

    def get_indexed_ds(self):
        return self.data_sources

    def generate_index(self, path):
        raise NotImplementedError('generate_index')

    def index_path(self, path: Optional[str] = None) -> None:
        if path is None:
            path = f'{self.get_gpfs_root()}/raw'

        super().index_path(path)

    def karabo(self, name):
        raise NotImplementedError('karabo')

    def suggest_index_path(self) -> str:
        return f'{self.get_gpfs_root()/usr/.xts_index.db'
