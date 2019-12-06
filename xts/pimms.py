
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


from typing import Optional

import os.path
import struct

import numpy

from xts import IndexedData, IndexedRoot


class PimmsData(IndexedData):
    def generate_index(self, raw_file_id, train_ids, pos):
        yield from zip(train_ids, pos)

    def walk_file(self, path, positions):
        with open(path, 'rb') as fp:
            for pos in positions:
                fp.seek(pos)

                n_rows, n_cols = struct.unpack('<ii', fp.read(8))
                yield numpy.fromfile(fp, numpy.uint16, n_rows * n_cols)


class PimmsRoot(IndexedRoot):
    def __init__(self, parent):
        prefix = 'pimms'

        if not isinstance(parent, str):
            try:
                # We don't check for the explicit type to avoid
                # importing the flash module only for this.
                self.root = f'{parent.get_gpfs_root()}/raw/pimms'
                prefix += f'_{parent.proposal}'

            except AttributeError:
                self.root = str(parent)
        else:
            self.root = str(parent)

        super().__init__(prefix)

        self._data = PimmsData(prefix)

    def schema(self):
        return super().schema(with_time=False, with_source=False)

    def data(self) -> PimmsData:
        return self._data

    def generate_index(self, path):
        # This function does pretty much all the heavylifting, while
        # PimmsData.generate_index() only relays the generated data.

        raw_tids = numpy.loadtxt(f'{path[:-4]}-settings.txt')[:, 0].astype(int)

        i = 0
        final_tids = []
        final_pos = []

        with open(path, 'rb') as fp:
            while True:
                header = fp.read(8)

                if not header:
                    break

                pos = fp.tell() - 8
                n_rows, n_cols = struct.unpack('<ii', header)
                fp.seek(pos + 8 + n_rows * n_cols * 2)

                train_id = raw_tids[i]
                i += 1

                # Beware, i is already advanced by one to cover the
                # possible continue statement below!

                # Rather crude code right now
                if train_id == 0:
                    if (raw_tids[i] - raw_tids[i-2]) == 2:
                        train_id = raw_tids[i] - 1
                    else:
                        train_id = raw_tids[i-2] + 1

                elif i < len(raw_tids) and train_id == raw_tids[i]:
                    if raw_tids[i-2] < train_id-1:
                        train_id -= 1
                    else:
                        print('duplicate without any gap detected')

                final_tids.append(train_id)
                final_pos.append(pos)

        yield final_tids, None, [self._data], dict(train_ids=final_tids,
                                                   pos=final_pos)

    def get_run_from_path(self, path):
        return int(os.path.basename(path)[:3])

    def index_path(self, path: Optional[str] = None) -> None:
        if path is None:
            path = f'{self.root}/*.bin'

        super().index_path(path)
