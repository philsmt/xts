
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


import os.path

import numpy as np
import h5py

from xts import HdfData, IndexedRoot, DataSource, DataRoot


def gen_bunch_ids(time_data):
    bunch_ids = time_data['seconds'].astype(np.uint64)
    bunch_ids *= 1000000000
    bunch_ids += time_data['nanoseconds']

    return bunch_ids


class LclsHdf5Source(HdfData):
    DATA_ROOT = 'Configure:0000/Run:0000/CalibCycle:0000'

    def __init__(self, prefix, source_path, dset_name):
        self.source_path = source_path

        super().__init__(prefix, f'{LclsHdf5Source.DATA_ROOT}/{source_path}/'
                                 f'{dset_name}')

    def generate_index(self, raw_file_id, h5in=None):
        time_data = np.array(h5in[f'{LclsHdf5Source.DATA_ROOT}/'
                                  f'{self.source_path}/time'])

        yield from zip(gen_bunch_ids(time_data), np.arange(time_data.shape[0]))


class LclsHdf5Root(IndexedRoot):
    TIMING_PATH = 'Configure:0000/Run:0000/CalibCycle:0000/EvrData' \
                  '::DataV4/NoDetector.0:Evr.0'

    def __init__(self, prefix):
        self.data_sources = []

        super().__init__(prefix)

    def source(self, source_path, dset_name):
        ds = LclsHdf5Source(self.prefix, source_path, dset_name)
        self.data_sources.append(ds)

        return ds

    def generate_index(self, path):
        with h5py.File(path, 'r') as h5in:
            time_data = np.array(h5in[LclsHdf5Root.TIMING_PATH]['time'])

            yield gen_bunch_ids(time_data), time_data['seconds'], \
                self.data_sources, dict(h5in=h5in)

    def get_run_from_path(self, path):
        basename = os.path.basename(path)

        run_pos = basename.index('r')

        if run_pos == -1:
            raise ValueError('invalid lcls filename')

        return int(basename[run_pos+1:-3])


class LclsXtcSource(DataSource):
    pass


class LclsXtcRoot(DataRoot):
    pass
