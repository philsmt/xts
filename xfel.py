
from typing import Optional

import h5py
import os.path
import re

import numpy

from xts import HdfData, IndexedRoot


class KaraboData(HdfData):
    def __init__(self, prefix, root, key):
        self.root = root

        super().__init__(prefix, f'{root}/{key}')

    def generate_index(self, raw_file_id, h5f=None, timing_train_ids=None):
        yield from zip(timing_train_ids, numpy.arange(len(timing_train_ids)))


class EuXfelRoot(IndexedRoot):
    def __init__(self, prefix: str) -> None:
        self.data_sources = []

        super().__init__(prefix)

    def get_run_from_path(self, path):
        basename = os.path.basename(path)

        da_pos = basename.index('-DA')

        if not basename.startswith('RAW-R') or da_pos == -1:
            raise ValueError('invalid karabo filename')

        return int(basename[5:da_pos])

    def karabo(self, root, key):
        ds = KaraboData(self.prefix, root, key)
        self.data_sources.append(ds)

        return ds

    def generate_index(self, path):
        with h5py.File(path, 'r') as h5f:
            timing = numpy.asarray(h5f['INDEX/trainId'])

            # List of all DataSource objects which apply to this data
            # aggregator.
            sources = [ds for ds in self.data_sources
                       if ds.root in {s.decode('ascii') for s
                                      in h5f['METADATA/dataSourceId'] if s}]

            if sources:
                yield timing, None, sources, dict(h5f=h5f,
                                                  timing_train_ids=timing)


class EuXfelProposal(EuXfelRoot):
    def __init__(self, instrument: str, cycle: int, proposal: int,
                 prefix: Optional[str] = None) -> None:
        self.instrument = instrument
        self.cycle = cycle
        self.proposal = proposal

        if prefix is None:
            prefix = f'exfel_{proposal}'

        super().__init__(prefix)

    @classmethod
    def auto_detect(cls, path: str = None) -> Optional['EuXfelProposal']:
        if path is None:
            path = os.path.realpath(os.getcwd())

        m = re.match('/gpfs/exfel/(exp|d/raw|u/scratch|u/usr)/(\w{3,6})/'
                     '(\d{6,6})/p(\d{6,6})', path)

        if m is not None:
            instrument, cycle_str, proposal_str = m.groups()
            return cls(instrument, int(cycle_str), int(proposal_str))

    def get_gpfs_root(self) -> str:
        return f'/gpfs/exfel/exp/{self.instrument}/{self.cycle}' \
               f'/p{self.proposal:06d}'

    def index_path(self, path: Optional[str] = None) -> None:
        if path is None:
            path = f'{self.get_gpfs_root()}/raw/r*/*'

        super().index_path(path)

    def suggest_index_path(self) -> str:
        return f'{self.get_gpfs_root()}/usr/.xts_index.db'
