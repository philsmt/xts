
from typing import Any, Callable, Dict, Generator, Iterable, List, Mapping, \
                   Optional, Tuple, Union, BinaryIO

from collections import OrderedDict as odict
import glob
import hashlib
import h5py
import importlib
import mmap
import multiprocessing
import multiprocessing.pool
import os.path
import queue
import sqlite3
import struct
import threading

import numpy


# wid: int, tid: int, *data: Any
TrainKernel = Callable[[int, int], Any]
# kernel: TrainKernel, target: TrainSet, *ds: DataSource, worker_id = 0
MapFunc = Callable


_INDEX_DATABASE_PATH = './xts_index.db'

try:
    mp_ctx = multiprocessing.get_context('fork')
except ValueError:
    DEFAULT_PL_METHOD = 'threading'
else:
    DEFAULT_PL_METHOD = 'processes'

DEFAULT_PL_WORKER = multiprocessing.cpu_count() // 3


class ShotId(int):
    pass


class TrainSet(object):
    def __init__(self, train_ids: Iterable[int]):
        self.train_ids = sorted(train_ids)

    def __str__(self):
        if len(self.train_ids) == 0:
            return 'TrainSet(empty)'
        else:
            return f'TrainSet({len(self.train_ids)} in ' \
                   f'[{self.train_ids[0]}, {self.train_ids[-1]}])'

    def __iter__(self):
        return iter(self.train_ids)

    def __len__(self):
        """Return the number of trains in this TrainSet."""

        return len(self.train_ids)

    def __eq__(self, y):
        """Return whether two TrainSets contain the same trains."""

        return self.train_ids == y.train_ids

    def __and__(self, y):
        """Return the intersection of two TrainSets."""

        return TrainSet([train_id for train_id in self.train_ids
                         if train_id in y.train_ids])

    def __or__(self, y):
        """Return the union of two TrainSets."""

        return TrainSet(sorted(list(set(self.train_ids + y.train_ids))))

    def __getitem__(self, index):
        return TrainSet(self.train_ids[index])

    def split(self, num):
        return [TrainSet(x) for x in numpy.array_split(self.train_ids, num)]

    def to_sql_str(self) -> str:
        return ','.join([str(train_id) for train_id in self.train_ids])

    def get_index_map(self) -> Dict[int, int]:
        return dict(zip(self.train_ids, range(len(self.train_ids))))

    def map_trains(self, kernel: TrainKernel, *ds: 'DataSource',
                   **kwargs) -> None:

        for val in ds:
            resolve_data_source(val).index_trains(self)

        map_kernel_by_train(kernel, self, *ds, **kwargs)

    def select_trains(self, kernel: TrainKernel, *ds: 'DataSource',
                      **kwargs) -> 'TrainSet':
        result_mask = alloc_array((len(self.train_ids),), dtype=bool,
                                  **kwargs)
        tid_map = self.get_index_map()

        def mask_kernel(wid, tid, *data):
            result_mask[tid_map[tid]] = kernel(wid, tid, *data)

        self.map_trains(mask_kernel, *ds, **kwargs)

        return TrainSet(list(numpy.array(self.train_ids)[result_mask]))

    def select_records(self, *args) -> 'TrainSet':
        tid_set = set(self.train_ids)

        for val in args:
            tid_set = tid_set.intersection(
                resolve_data_source(val).walk_records(self)
            )

        return TrainSet(tid_set)

    def map_data(self, *ds: 'DataSource', **kwargs) -> Tuple[numpy.ndarray]:
        shapes = dtypes = None

        def shape_kernel(wid, tid, *data):
            nonlocal shapes, dtypes
            shapes = [d.shape for d in data]
            dtypes = [d.dtype for d in data]

        self[:1].map_trains(shape_kernel, *ds, **kwargs)

        tid_map = self.get_index_map()
        arrays = [alloc_array((len(self), *shapes[i]), dtypes[i], per_worker=False, **kwargs)
                  for i in range(len(ds))]

        def fill_kernel(wid, tid, *data):
            tid_idx = tid_map[tid]

            for i, d in enumerate(data):
                if d is not None:
                    arrays[i][tid_idx] = d

        self.map_trains(fill_kernel, *ds, **kwargs)

        return tuple(arrays)


class TrainRange(TrainSet):
    def __init__(self, min_tid=None, max_tid=None, inclusive=True):
        # Should use ALL data roots

        eq_sign = '=' if inclusive else ''
        where_clauses = []

        if min_tid is not None:
            where_clauses.append(f'train_id >{eq_sign} {min_tid}')

        if max_tid is not None:
            where_clauses.append(f'train_id <{eq_sign} {max_tid}')

        where_str = ' AND '.join(where_clauses) if where_clauses \
                    else '1'

        super().__init__([res['train_id'] for res in
                          index_dbc().execute(f'''
                              SELECT train_id
                              FROM trains
                              WHERE {where_str}
                          ''')])


class TimeRange(TrainSet):
    def __init__(self, min_time=None, max_time=None):
        pass


class Run(TrainSet):
    def __init__(self, *args, **kwargs):
        run_strs = []
        tid_sets = []

        if kwargs:
            for key, value in kwargs.items():
                try:
                    root = getattr(env, key)
                except KeyError:
                    raise ValueError(
                        f'data root \'{key}\' not defined in environment'
                    ) from None

                tid_sets.append(set(root.by_run(value)))
                run_strs.append(f'{key}={value}')

        elif args:
            roots = [dr for dr in env.__dict__.values()
                     if isinstance(dr, DataRoot)]

            if len(args) > len(roots):
                raise ValueError('insufficient number of data roots defined '
                                 'in environment')

            for root, value in zip(roots, args):
                tid_sets.append(set(root.by_run(value)))
                run_strs.append(f'{root}={value}')

        if not tid_sets:
            raise ValueError('no run contraints given')

        final_set = set()

        for _set in tid_sets:
            final_set |= _set

        super().__init__(final_set)

        self.run_str = ', '.join(run_strs)

    def __str__(self):
        if len(self.train_ids) == 0:
            # Kind of impossible in practice, but this method should
            # not throw exceptions for otherwise valid object states.

            trains_str = '0 trains'
        else:
            trains_str = f'{len(self.train_ids)} trains ' \
                         f'[{self.train_ids[0]}, {self.train_ids[-1]}]'

        return f'Run({trains_str} in {self.run_str})'


class DataSource(object):
    def index_trains(self, target: TrainSet) -> None:
        '''
        Index this data source for a TrainSet.

        This method gives the DataSource a chance to build an index, typically
        prior to a mapping operation. It is not required and may be skipped.

        DO NOT build an index in any method of the walk_ family! They are run
        in parallel any will run into race conditions when writing to the index
        database.
        '''

        pass

    def walk_trains(self, target: TrainSet) -> Generator:
        # Walk actual data

        raise NotImplementedError('walk_trains')

    def walk_records(self, target: TrainSet) -> Generator:
        # Walk existing records, i.e. train IDs with data.

        raise NotImplementedError('walk_records')


class ArrayData(DataSource):
    def __init__(self, train_ids: Iterable[int], data: Iterable[Any]) -> None:
        if len(train_ids) != len(data):
            raise ValueError('arguments mismatch in their length')

        self.train_ids = train_ids
        self.index = dict(zip(train_ids, range(len(train_ids))))
        self.data = data

    def walk_trains(self, target: TrainSet) -> Generator:
        for train_id in target:
            try:
                idx = self.index[train_id]
            except KeyError:
                yield None
            else:
                yield self.data[idx]

    def walk_records(self, target: TrainSet) -> Generator:
        for train_id in target:
            if train_id in self.index:
                yield train_id


class MappedData(DataSource):
    def __init__(self, data_map: Mapping[int, Any]) -> None:
        self.data_map = data_map

    def walk_trains(self, target: TrainSet) -> Generator:
        for train_id in target:
            try:
                yield self.data_map[train_id]
            except KeyError:
                yield None

    def walk_records(self, target: TrainSet) -> Generator:
        for train_id in target:
            if train_id in self.data_map:
                yield train_id


class IndexedData(DataSource):
    def __init__(self, prefix):
        self.prefix = prefix

    @property
    def source_cond(self):
        try:
            return self._source_cond
        except AttributeError:
            try:
                source_id = self.get_source_id()
            except NotImplementedError:
                self._source_cond = ''
            else:
                self._source_cond = f'AND source_id = {source_id}'

            return self._source_cond

    @property
    def source_insert(self):
        try:
            return self._source_insert
        except AttributeError:
            try:
                source_id = self.get_source_id()
            except NotImplementedError:
                self._source_insert = '', ''
            else:
                self._source_insert = ' source_id,', f' {source_id},'

            return self._source_insert

    def get_source_id(self) -> int:
        raise NotImplementedError('get_source_id')

    def get_records_cursor(self, tid_list_str):
        return index_dbc().execute(f'''
            SELECT train_id, file_id
            FROM {self.prefix}_records
            WHERE train_id IN ({tid_list_str})
                  {self.source_cond}
            ORDER BY train_id ASC
        ''')

    def get_file_generator(self, file_id, tid_list_str):
        dbc = index_dbc()

        file_row = dbc.execute(f'''
            SELECT path
            FROM {self.prefix}_files
            WHERE file_id = {file_id}
        ''').fetchone()

        records_cursor = dbc.execute(f'''
            SELECT position
            FROM {self.prefix}_records
            WHERE train_id IN ({tid_list_str}) AND
                  file_id = {file_id}
                  {self.source_cond}
            ORDER BY train_id ASC
        ''')

        records_rows = [row[0] for row in records_cursor]

        return self.walk_file(file_row['path'], records_rows), \
            len(records_rows)

    def generate_index(self, raw_file_id):
        raise NotImplementedError('generate_index')

    def walk_file(self, path: str, positions: Iterable[int]) -> Generator:
        raise NotImplementedError('walk_file')

    def walk_trains(self, target: TrainSet, *args, **kwargs) -> Generator:
        tid_list_str = target.to_sql_str()

        records_cursor = self.get_records_cursor(tid_list_str, *args,
                                                 **kwargs)

        target_it = iter(target)
        gen_it = {}
        gen_len = {}

        for train_id, row in zip(target_it, records_cursor):
            while train_id != row['train_id']:
                # Skip this train if it's not in our index
                yield None
                train_id = next(target_it)

            file_id = row['file_id']

            try:
                gen = gen_it[file_id]
            except KeyError:
                gen_it[file_id], gen_len[file_id] = \
                    self.get_file_generator(file_id, tid_list_str)
                gen = gen_it[file_id]

            yield next(gen)

            gen_len[file_id] -= 1
            if gen_len[file_id] == 0:
                try:
                    next(gen)
                except StopIteration:
                    pass

    def walk_records(self, target: TrainSet) -> Generator:
        records_cursor = index_dbc().execute(f'''
            SELECT train_id
            FROM {self.prefix}_records
            WHERE train_id IN ({target.to_sql_str()})
                  {self.source_cond}
            ORDER BY train_id ASC
        ''')

        for row in records_cursor:
            yield row[0]

    def index(self, raw_file_id: int, *args, **kwargs) -> None:
        index_gen = self.generate_index(raw_file_id, *args, **kwargs)

        source_col, source_val = self.source_insert

        index_dbc().executemany(f'''
            INSERT OR IGNORE INTO {self.prefix}_records (
                train_id, {source_col} file_id, position
            )
            VALUES (?, {source_val} {raw_file_id}, ?)
        ''', index_gen)


class PackedData(IndexedData):
    def walk_packed_file(self, path: str,
                         positions: Iterable[int]) -> Generator:
        with open(path, 'rb') as fp:
            self.read_packed_header(fp)

            for pos in positions:
                fp.seek(pos)
                yield self.read_packed_entry(fp)

    def read_packed_header(self, fp: BinaryIO) -> None:
        pass

    def read_packed_entry(self, fp: BinaryIO) -> Any:
        raise NotImplementedError('decode_packed_entry')

    def get_records_cursor(self, tid_list_str, force_type=None):
        if force_type in ('raw', 'packed'):
            file_id_col = force_type + '_file_id'
        else:
            file_id_col = 'ifnull(packed_file_id, raw_file_id)'

        return index_dbc().execute(f'''
            SELECT train_id, {file_id_col} AS file_id
            FROM {self.prefix}_records
            WHERE train_id IN ({tid_list_str})
                  {self.source_cond}
            ORDER BY train_id ASC
        ''')

    def get_file_generator(self, file_id, tid_list_str):
        dbc = index_dbc()

        file_row = dbc.execute(f'''
            SELECT path, type
            FROM {self.prefix}_files
            WHERE file_id = {file_id}
        ''').fetchone()

        file_type = file_row['type']

        records_cursor = dbc.execute(f'''
            SELECT {file_type}_position
            FROM {self.prefix}_records
            WHERE train_id IN ({tid_list_str}) AND
                  {file_type}_file_id = {file_id}
                  {self.source_cond}
            ORDER BY train_id ASC
        ''')

        records_rows = [row[0] for row in records_cursor]

        if file_type == 'packed':
            walk_func = self.walk_packed_file
        else:
            walk_func = self.walk_file

        return walk_func(file_row['path'], records_rows), len(records_rows)

    # Generic
    def index(self, raw_file_id: int, *args, **kwargs) -> None:
        index_gen = self.generate_index(raw_file_id,
                                        self.packed_root is not None,
                                        *args, **kwargs)

        dbc = index_dbc()
        source_col, source_val = self.source_insert

        if self.packed_root is not None:
            packed_hash = hashlib.sha1()

            for val in self.generate_packed_name(raw_file_id, *args, **kwargs):
                packed_hash.update(val)

            packed_path = '{0}/{1}.xts'.format(self.packed_root,
                                               packed_hash.hexdigest())

            file_row = dbc.execute(f'''
                SELECT file_id
                FROM {self.prefix}_files
                WHERE path = "{packed_path}" AND type = "packed"
            ''').fetchone()

            if file_row is None:
                dbc.execute(f'''
                    INSERT INTO {self.prefix}_files (path, type)
                    VALUES ("{packed_path}", "packed")
                ''')

                file_row = dbc.execute(f'''
                    SELECT last_insert_rowid()
                    FROM {self.prefix}_files
                ''').fetchone()

            packed_file_id = file_row[0]

            train_ids = []
            raw_pos = []
            packed_pos = []

            try:
                header_args = next(index_gen)
            except StopIteration:
                return

            with open(packed_path, 'wb') as fp:
                self.write_packed_header(fp, *header_args)

                for train_id, pos, data in index_gen:
                    train_ids.append(train_id)
                    raw_pos.append(pos)
                    packed_pos.append(self.write_packed_entry(fp, data))

            dbc.executemany(f'''
                INSERT OR IGNORE INTO {self.prefix}_records (
                    train_id, {source_col} raw_file_id, raw_position,
                    packed_file_id, packed_position
                )
                VALUES (?, {source_val} {raw_file_id}, ?, {packed_file_id}, ?)
            ''', zip(train_ids, raw_pos, packed_pos))
        else:
            dbc.executemany(f'''
                INSERT OR IGNORE INTO {self.prefix}_records (
                    train_id, {source_col} raw_file_id, raw_position
                )
                VALUES (?, {source_val} {raw_file_id}, ?)
            ''', index_gen)


class PackedArrayData(PackedData):
    HEADER_MAGIC = b'XTSPAK-ARRAY'
    FORMAT_VERSION = 1

    def write_packed_header(self, fp: BinaryIO, data: numpy.ndarray):
        fp.write(PackedArrayData.HEADER_MAGIC)
        fp.write(PackedArrayData.FORMAT_VERSION.to_bytes(4, 'little'))

        type_str = data.dtype.str.encode('ascii')
        fp.write(type_str)
        fp.write(b'\x00' * (4 - len(type_str)))

        fp.write(struct.pack('<B', len(data.shape)))

    def write_packed_entry(self, fp: BinaryIO, data: numpy.ndarray):
        offset = fp.tell()

        fp.write(struct.pack('<' + len(data.shape) * 'I', *data.shape))
        data.tofile(fp)

        return offset

    def read_packed_header(self, fp: BinaryIO):
        if fp.read(len(PackedArrayData.HEADER_MAGIC)) != \
                PackedArrayData.HEADER_MAGIC:
            raise ValueError('unknown file format')

        version = int.from_bytes(fp.read(4), 'little')

        if version > PackedArrayData.FORMAT_VERSION:
            raise ValueError('incompatible file format')

        self._dtype = numpy.dtype(fp.read(4).rstrip(b'\x00'))
        self._rank = fp.read(1)[0]

        self._fmt = '<' + self._rank * 'I'

    def read_packed_entry(self, fp: BinaryIO):
        shape = struct.unpack(self._fmt, fp.read(self._rank*4))

        elements = 1
        for _s in shape:
            elements *= _s

        return numpy.fromfile(fp, self._dtype, elements).reshape(shape)


class HdfData(IndexedData):
    def __init__(self, prefix, hdf_path):
        super().__init__(prefix)

        self.hdf_path = hdf_path

    def get_source_id(self):
        dbc = index_dbc()

        source_row = dbc.execute(f'''
            SELECT source_id
            FROM {self.prefix}_sources
            WHERE name = "{self.hdf_path}"
        ''').fetchone()

        if source_row is None:
            dbc.execute(f'''
                INSERT INTO {self.prefix}_sources (name)
                VALUES ("{self.hdf_path}")
            ''')

            source_row = dbc.execute(f'''
                SELECT last_insert_rowid()
                FROM {self.prefix}_sources
            ''').fetchone()

        return source_row[0]

    def generate_packed_name(self, raw_file_id, *args, **kwargs):
        yield self.prefix.encode('ascii')
        yield raw_file_id.to_bytes(4, 'little')
        yield self.hdf_path.encode('ascii')

    def walk_file(self, path, positions):
        with h5py.File(path, 'r') as h5f:
            data = numpy.array(h5f[self.hdf_path])

            for pos in positions:
                yield data[pos]

    def dtype(self) -> numpy.dtype:
        pass

    def shape(self, train_id: Optional[int] = None) -> Tuple[int, ...]:
        pass


class DataRoot(object):
    def by_run(self, run_id: int) -> Iterable[int]:
        raise NotImplementedError('DataRoot implementation does not support '
                                  'grouping of trains by a run')

    def suggest_index_path(self) -> str:
        raise NotImplementedError('DataRoot implementation does not provide '
                                  'a suggested path for the index database')

    def schema(sql):
        return None


class IndexedRoot(DataRoot):
    def __init__(self, prefix):
        self.prefix = prefix

    def schema(self, with_time=True, with_source=True):
        tables = odict(
            trains=odict(
                train_id='INTEGER PRIMARY KEY NOT NULL',
                run_id='INTEGER NOT NULL',
                time='INTEGER NULL',
                __indices__={'run': ['run_id']}
            ),

            records=odict(
                train_id='INTEGER NOT NULL',
                source_id='INTEGER NOT NULL',
                file_id='INTEGER NOT NULL',
                position='INTEGER NOT NULL',
                __primary_key__=['train_id', 'source_id']
            ),

            files=odict(
                file_id='INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL',
                path='TEXT'
            ),

            sources=odict(
                source_id='INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL',
                name='TEXT'
            )
        )

        if not with_time:
            del tables['trains']['time']

        if not with_source:
            del tables['records']['source_id']
            del tables['records']['__primary_key__'][1]
            del tables['sources']

        return tables

    def get_indexed_ds(self):
        raise StopIteration

    def by_run(self, run_id: int) -> List[int]:
        # Assume aot indexing for now
        # It is here where we could index just-in-time

        train_cursor = index_dbc().execute(f'''
            SELECT train_id
            FROM {self.prefix}_trains
            WHERE run_id = {run_id}
        ''')

        return [row['train_id'] for row in train_cursor]

    def register_file(self, path) -> int:
        dbc = index_dbc()

        dbc.execute(f'''
            INSERT INTO {self.prefix}_files (path)
            VALUES ("{path}")
        ''')

        file_id = dbc.execute(f'''
            SELECT last_insert_rowid()
            FROM {self.prefix}_files
        ''').fetchone()[0]

        return file_id

    def index_file(self, path) -> None:
        dbc = index_dbc()

        file_row = dbc.execute(f'''
            SELECT file_id
            FROM {self.prefix}_files
            WHERE path = "{path}"
        ''').fetchone()

        if file_row is not None:
            return

        file_id = self.register_file(path)
        run_id = self.get_run_from_path(path)

        index_gen = self.generate_index(path)

        try:
            train_ids, times, data_sources, ds_kwargs = next(index_gen)
        except StopIteration:
            return

        for ds in data_sources:
            ds.index(file_id, **ds_kwargs)

        if times is not None:
            dbc.executemany(f'''
                INSERT OR IGNORE INTO {self.prefix}_trains (
                    train_id, run_id, time
                )
                VALUES (?, {run_id}, ?)
            ''', zip(train_ids, times))
        else:
            dbc.executemany(f'''
                INSERT OR IGNORE INTO {self.prefix}_trains (
                    train_id, run_id
                )
                VALUES (?, {run_id})
            ''', zip(train_ids))

    def index_path(self, path: str) -> None:
        for path in glob.iglob(path):
            self.index_file(path)

        index_dbc().commit()


class PackedRoot(IndexedRoot):
    def schema(self, *args, **kwargs):
        tables = super().schema(*args, **kwargs)

        del tables['records']['file_id']
        del tables['records']['position']

        tables['records'].update(
            raw_file_id='INTEGER NOT NULL', raw_position='INTEGER NOT NULL',
            packed_file_id='INTEGER NULL', packed_position='INTEGER NULL'
        )

        tables['files']['type'] = 'TEXT'

        return tables

    def register_file(self, path, type_='raw') -> int:
        dbc = index_dbc()

        dbc.execute(f'''
            INSERT INTO {self.prefix}_files (path, type)
            VALUES ("{path}", "{type_}")
        ''')

        file_id = dbc.execute(f'''
            SELECT last_insert_rowid()
            FROM {self.prefix}_files
        ''').fetchone()[0]

        return file_id


class Environment(object):
    def __call__(self, path: str = None, **kwargs):
        if path is not None:
            if os.path.isdir(path):
                path = os.path.join(path, '.xtsenv.py')

            module_name = os.path.basename(path)[:path.rfind('.')]

            if module_name == '.xtsenv':
                module_name = '__hidden_xtsenv__'

            try:
                env_mod = importlib.util.spec_from_file_location(
                    'xts.env.' + module_name, path
                ).loader.load_module()
            except ImportError:
                return

            else:
                for key in dir(env_mod):
                    value = getattr(env_mod, key)

                    if isinstance(value, DataRoot) or \
                            isinstance(value, DataSource):
                        setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)


def parallelized(func: Callable):
    def parallelized_func(kernel: TrainKernel, target: TrainSet,
                          *args, **kwargs):
        if 'worker_id' in kwargs:
            raise ValueError('worker_id may not be used as a keyword '
                             'argument for parallelized functions')

        pl_worker = get_pl_worker(kwargs)
        pl_method = get_pl_method(kwargs)

        if pl_worker == 1:
            func(kernel, target, *args, worker_id=0, **kwargs)
            return

        if pl_method == 'processes':
            if _INDEX_DATABASE_PATH == ':memory:':
                raise ValueError('in-memory database is not supported '
                                 'with \'processes\' parallelization method')

            queue_class = mp_ctx.Queue
            pool_class = mp_ctx.Pool
            init_func = _worker_process_init
            run_func = _worker_process_run

        elif pl_method == 'threads':
            queue_class = queue.Queue
            pool_class = multiprocessing.pool.ThreadPool
            init_func = _worker_thread_init
            run_func = _worker_thread_run

            global _worker_id_map, _worker_func, _worker_args, \
                _worker_kwargs, _kernel
            _worker_id_map = {}
            _worker_func = func
            _worker_args = args
            _worker_kwargs = kwargs
            _kernel = kernel

        id_queue = queue_class()
        for worker_id in range(pl_worker):
            id_queue.put(worker_id)

        init_args = (id_queue, func, kernel, args, kwargs)

        with pool_class(pl_worker, init_func, init_args) as p:
            p.map(run_func, target.split(pl_worker))

    return parallelized_func


def get_pl_worker(kwargs: Dict[str, Any], default: Optional[str] = None) -> int:
    try:
        pl_worker = int(kwargs['pl_worker'])
    except KeyError:
        if default is not None:
            pl_worker = default
        else:
            pl_worker = DEFAULT_PL_WORKER

    except ValueError:
        raise ValueError('invalid pl_worker value') from None

    else:
        del kwargs['pl_worker']

    return pl_worker


def get_pl_method(kwargs: Dict[str, Any], default: Optional[str] = None) -> str:
    try:
        pl_method = kwargs['pl_method']
    except KeyError:
        if default is not None:
            pl_method = default
        else:
            pl_method = DEFAULT_PL_METHOD

    else:
        del kwargs['pl_method']

    if pl_method not in ('processes', 'threads'):
        raise ValueError('invalid parallelization method')

    return pl_method


def get_pl_env(kwargs: Dict[str, Any], pl_worker: Optional[str] = None,
               pl_method: Optional[str] = None) -> Tuple[int, str]:
    return get_pl_worker(kwargs, pl_worker), get_pl_method(kwargs, pl_method)


def alloc_array(shape: Tuple[int], dtype: numpy.dtype,
                per_worker: bool = False, **kwargs) -> numpy.ndarray:
    pl_method = get_pl_method(kwargs)

    if per_worker:
        pl_worker = get_pl_worker(kwargs)
        shape = (pl_worker,) + shape

    if pl_method == 'processes':
        n_elements = 1
        for _s in shape:
            n_elements *= _s

        n_bytes = n_elements * numpy.dtype(dtype).itemsize
        n_pages = n_bytes // mmap.PAGESIZE + 1

        buf = mmap.mmap(-1, n_pages * mmap.PAGESIZE,
                        flags=mmap.MAP_SHARED | mmap.MAP_ANONYMOUS,
                        prot=mmap.PROT_READ | mmap.PROT_WRITE)
        return numpy.frombuffer(memoryview(buf)[:n_bytes],
                                dtype=dtype).reshape(shape)

    elif pl_method == 'threads':
        return numpy.zeros(shape, dtype=dtype)


def _worker_process_init(id_queue: multiprocessing.Queue,
                         func: MapFunc, kernel: TrainKernel,
                         args: Tuple[Any, ...], kwargs: Dict[str, Any]):
    global _worker_id, _worker_func, _worker_args, _worker_kwargs, _kernel
    _worker_id = id_queue.get()
    _worker_func = func
    _worker_args = args
    _worker_kwargs = kwargs
    _kernel = kernel

    global _INDEX_DATABASE_CONN
    del _INDEX_DATABASE_CONN


def _worker_process_run(target: TrainSet) -> None:
    _worker_func(_kernel, target, *_worker_args,
                 worker_id=_worker_id, **_worker_kwargs)


def _worker_thread_init(id_queue: queue.Queue,
                        func: MapFunc, kernel: TrainKernel,
                        args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
    global _worker_id_map
    _worker_id_map[threading.get_ident()] = id_queue.get()


def _worker_thread_run(target: TrainSet):
    _worker_func(_kernel, target, *_worker_args,
                 worker_id=_worker_id_map[threading.get_ident()],
                 **_worker_kwargs)


@parallelized
def map_kernel_by_train(kernel: TrainKernel, target: TrainSet, *data_sources,
                        worker_id=0, **kwargs) -> None:
    # DO NOT CALL DIRECTLY! The data generators need to be prepared!

    ds_gens = get_data_generators(target, data_sources, kwargs)

    for train_id, *ds_data in zip(target, *ds_gens):
        kernel(worker_id, train_id, *ds_data)


def resolve_data_source(val) -> DataSource:
    if isinstance(val, str):
        try:
            ds = getattr(env, val)
        except AttributeError:
            raise ValueError(f'data source \'{val}\' not defined in current '
                             f'environment') \
                from None
    elif isinstance(val, DataSource):
        ds = val
    else:
        raise ValueError(f'invalid value passed as data source')

    return ds


def get_data_generators(target: TrainSet, data_sources, kwargs) -> List[Generator]:
    gens = []

    for val in data_sources:
        ds = resolve_data_source(val)

        if isinstance(val, str) and ('_' + val) in kwargs:
            gen = ds.walk_trains(target, **kwargs['_' + val])
        else:
            gen = ds.walk_trains(target)

        gens.append(gen)

    return gens


def index_opts(strategy: str = 'jit',
               path: Union[str, DataRoot, None] = _INDEX_DATABASE_PATH):
    try:
        _initialized
    except NameError:
        pass
    else:
        raise RuntimeError('xts is already initialized')

    # if strategy not in ('aot', 'jit'):
    if strategy not in ('aot'):
        raise ValueError('index strategy not supported')

    if path is None:
        path = ':memory:'

        global DEFAULT_PL_METHOD
        DEFAULT_PL_METHOD = 'threads'

    elif isinstance(path, DataRoot):
        path = path.suggest_index_path()

    global _INDEX_DATABASE_PATH
    _INDEX_DATABASE_PATH = path


def get_index_path():
    return _INDEX_DATABASE_PATH


def build_schema_sql(tables, prefix) -> str:
    sql = ''

    for table_name, table_def in tables.items():
        sql += f'CREATE TABLE IF NOT EXISTS {prefix}_{table_name} (\n'

        col_sql = []

        for col_name, col_def in table_def.items():
            if not col_name.startswith('__'):
                col_sql.append(f'\t{col_name} {col_def}')

        try:
            primary_key = table_def['__primary_key__']
        except KeyError:
            pass
        else:
            col_sql.append(f'\tPRIMARY KEY({", ".join(primary_key)})')

        sql += ',\n'.join(col_sql)
        sql += '\n);\n'

        try:
            indices = table_def['__indices__']
        except KeyError:
            pass
        else:
            for index_name, index_columns in indices.items():
                sql += f'CREATE INDEX IF NOT EXISTS {prefix}_{table_name}_{index_name} ' \
                       f'ON {prefix}_{table_name} ({", ".join(index_columns)});\n'

        sql += '\n'

    return sql


def index_dbc():
    global _INDEX_DATABASE_CONN

    try:
        return _INDEX_DATABASE_CONN
    except NameError:
        dbc = sqlite3.connect(_INDEX_DATABASE_PATH, check_same_thread=False)
        dbc.row_factory = sqlite3.Row

        for type_ in [numpy.int8, numpy.uint8, numpy.int16, numpy.uint16,
                      numpy.int32, numpy.uint32, numpy.int64, numpy.uint64]:
            sqlite3.register_adapter(type_, int)

        for type_ in [numpy.float32, numpy.float64]:
            sqlite3.register_adapter(type_, float)

        dbc.execute('PRAGMA journal_mode=WAL')

        for root in [dr for dr in env.__dict__.values()
                     if isinstance(dr, DataRoot)]:
            tables = root.schema()

            if tables is None:
                continue

            dbc.executescript(build_schema_sql(tables, root.prefix))

        dbc.commit()

        global _initialized
        _initialized = True
        _INDEX_DATABASE_CONN = dbc

        return dbc


env = Environment()
