
from typing import Optional

import h5py
import os
import re

import numpy

from xts import PackedArrayData, HdfData, PackedRoot


class FlashHdfData(PackedArrayData, HdfData):
    # Custom implementation of HdfData, as we have to deal with the
    # overlap problem...

    def __init__(self, prefix, group, dset_act, dset_validate,
                 packed_root=None):
        super().__init__(prefix, f'{group}/{dset_act}')

        self.validate_path = f'{group}/{dset_act}{dset_validate}'
        self.packed_root = packed_root

    # h5f, timing_train_ids should be keyword arguments, this generator
    # remains compatible with IndexedData
    def generate_index(self, raw_file_id, is_packed=False, h5f=None,
                       timing_train_ids=None):
        try:
            validate_h5d = h5f[self.validate_path]
        except KeyError:
            return
        else:
            if len(validate_h5d.shape) > 1 and validate_h5d.shape[1] > 1:
                raise ValueError('validation dataset has too many dimensions')

            validate_data = numpy.asarray(validate_h5d).flatten()

        if numpy.issubdtype(validate_data.dtype, numpy.floating):
            train_mask = numpy.isfinite(validate_data)
        elif numpy.issubdtype(validate_data.dtype, numpy.integer):
            train_mask = validate_data != 0
        else:
            raise ValueError(
                f'unable to check dtype \'{validate_data.dtype}\' of '
                f'validation dataset for \'{self.act_path}\''
            )

        act_train_ids = timing_train_ids[train_mask]
        raw_pos = numpy.arange(timing_train_ids.shape[0])[train_mask]

        if is_packed:
            data = numpy.asarray(h5f[self.act_path])[train_mask]

            yield (data[0],)
            yield from zip(act_train_ids, raw_pos, data)

        else:
            yield from zip(act_train_ids, raw_pos)


class FlashRoot(PackedRoot):
    def __init__(self, prefix: str, user: str,
                 packed_root: str = './') -> None:
        self.user = user
        self.packed_root = packed_root

        self.data_sources = []

        super().__init__(prefix)

    def get_run_from_path(self, path):
        basename = os.path.basename(path)

        run_pos = basename.index('run')
        file_pos = basename.index('file')

        if run_pos == -1 or file_pos == -1:
            raise ValueError('invalid flash filename')

        return basename[run_pos+3:file_pos-1]

    def get_packed_root(self) -> str:
        return self.packed_root

    def get_timing_dset(self) -> str:
        return f'Timing/time stamp/{self.user}'

    def hdf(self, group, dset, overlap_suffix='',
            packed=False) -> FlashHdfData:
        ds = FlashHdfData(self.prefix, group, dset, overlap_suffix,
                          self.get_packed_root() if packed else None)
        self.data_sources.append(ds)

        return ds

    def generate_index(self, path):
        with h5py.File(path, 'r') as h5f:
            timing = numpy.asarray(h5f[self.get_timing_dset()])

            yield timing[:, 2], timing[:, 0], self.data_sources, \
                dict(h5f=h5f, timing_train_ids=timing[:, 2])


class FlashProposal(FlashRoot):
    FLASH1_BEAMLINES = ('bl1', 'bl2', 'bl3', 'pg1', 'pg2', 'thz')
    FLASH2_BEAMLINES = ('fl21', 'fl22', 'fl23', 'fl24', 'fl25', 'fl26')

    def __init__(self, beamline: str, year: int, proposal: int,
                 user: Optional[str] = None, prefix: Optional[str] = None,
                 packed_root: Optional[str] = None) -> None:
        self.beamline = beamline
        self.year = year
        self.proposal = proposal

        if user is None:
            if self.beamline in FlashProposal.FLASH1_BEAMLINES:
                user = 'fl1user1'
            elif self.beamline in FlashProposal.FLASH2_BEAMLINES:
                user = 'fl2user1'
            else:
                raise ValueError('could not detect user based on beamline')

        if prefix is None:
            prefix = f'flashU_{proposal}'

        if packed_root is None:
            packed_root = self.get_gpfs_root() + '/scratch_cc/xts_packed'

            if not os.path.isdir(packed_root):
                os.mkdir(packed_root)

        self.packed_root = packed_root

        super().__init__(prefix, user, packed_root)

    @classmethod
    def auto_detect(cls, path: str = None) -> Optional['FlashProposal']:
        if path is None:
            path = os.path.realpath(os.getcwd())

        m = re.match('/asap3/flash/gpfs/(\w{3,4})/(\d{4,4})/data/(\d{8,8})',
                     path)

        if m is not None:
            beamline, year_str, proposal_str = m.groups()
            return cls(beamline, int(year_str), int(proposal_str))

    def get_gpfs_root(self) -> str:
        return f'/asap3/flash/gpfs/{self.beamline}/{self.year}' \
               f'/data/{self.proposal}'

    def index_path(self, path: Optional[str] = None) -> None:
        if path is None:
            path = f'{self.get_gpfs_root()}/raw/hdf/*/{self.user}/*'

        super().index_path(path)

    def suggest_index_path(self) -> str:
        return f'{self.get_gpfs_root()}/processed/.xts_index.db'


class FlashMachine(FlashProposal):
    # Slightly modified FlashProposal to accommodate some slight path
    # variations. It is a separate DataRoot as it will cover the same
    # trains in completely different runs.

    def __init__(self, parent: FlashProposal,
                 prefix: Optional[str] = None) -> None:
        if prefix is None:
            prefix = f'flashM_{parent.proposal}'

        super().__init__(parent.beamline, parent.year, parent.proposal,
                         user='pbd', prefix=prefix)

    def get_timing_dset(self) -> str:
        return 'Timing/time stamp/gmd'
