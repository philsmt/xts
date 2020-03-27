# xts

xts (eXtensible Toolkit for Shot analysis) provides a way to select, match, filter and process sets of event-based data, i.e. FEL trains or laser shots, across multiple backends at the same time - without any need for manual I/O as well as automatic scaling across any number of cores. With a thin layer of facility-specific code written once to specify how to index and read raw data, the location and format of the data is invisible and irrelevant during the actual data analysis. Its interface is based around map and/or reduce patterns for parallel processing of individal data items.


## Quick guide
Individual events (called _pulses_) may be spaced homogeneously or in groups (called _trains_), which themselves appear at regular intervals. This separation allows an efficient representation of a multitude of data structures, e.g. the burst mode in use at the FLASH and European XFEL facilities. 

Currently, only the access to trains is fully implemented. In the absence of this groping, trains may be treated as groups containing only one event. The fundamental object is `TrainSet`, which is an abstraction for an ordered list of train IDs uniquely identifying an event across all used backends. These backend are then able to retrieve the record for a particular data source at this train ID.

The data is typically not accessed directly, but via map patterns that operate only on a single record at a time. The sample below reads in all electron VMI images normalized by GMD intensity (_not actually physically meaningful!_).

```python
import xts

# Load the configuration file that defines the environment of this beamtime.
xts.env('~/flash_chiral')

# The run object initially contains all trains belonging to this run, while
# select_records('eVMI') selects # only those where actual data for the passed
# data sources is present (in this case, the electron VMI). Run is a special
# subclass of TrainSet allowing addressing by run number, while
# select_records(*ds) returns just a TrainSet, which is a subset of the original
# run. If this is not performed, the kernel may be called with None for some
# data sources.
run = xts.Run(flash=25367).select_records('eVMI', 'gmd')

# alloc_array reserves memory and returns a numpy array for use in the kernels.
# Depending upon the parallelization configuration, this may be allocated in a
# shared memory region or even subdivided into further arrays for each worker
# (e.g. to parallelize a reduce operation).
evmi_data = xts.alloc_array((len(run), 1280, 960), np.float32, per_worker=False)

# Returns a dictionary mapping trainIds to buffer positions in the same order.
tid_map = run.get_index_map()

# The kernel acts on each train individually and is passed a workerId, trainId
# and the data sources requested.
# The access to evmi_data and tid_map is provided through closures.
def read_kernel(wid, tid, eVMI_data, gmd_data):
    evmi_data[tid_map[tid]] = eVMI_data / gmd_data

# Perform the map operation.
run.map_trains(read_kernel, 'eVMI', 'gmd')

# evmi_data now contains the data. By default, it is parallelized via several
# worker processes proportional to the total number of cores present. It the kernel
# releases the GIL itself, the method may also be changed to threads.
```

The configuration file contains the specific environment of a beamtime. It is defined by a collection of `DataRoot` objects and `DataSource` objects. A `DataRoot` defines the way data is organized by train ID, while the `DataSource` is then responsible for actually loading data for a specific train ID. In order to considerably speed up the reading process, data may be indexed (and packed into a highly efficient data format) transparently. This is implemented via subclasses of `DataRoot`/`DataSource`, which may instead by extended to gain this functionality. In particular for FLASH, where any read access to a file (even if only for a single train) requires around 1-2s of decompression, this can speed up the I/O by a factor of 50-100x. The configuration for the sample above is:

```python
import xts

from xts.flash import FlashProposal, FlashMachine
from xts.pimms import PimmsRoot

# Define the DataRoot for the data recorded by the FLASH user DAQ. If called
# within the GPFS path of this proposal, it may also be created by
# flash = FlashProposal.auto_detect()
flash = FlashProposal(year=2018, beamline='bl1', proposal=11004732)

# Define a number of data sources by HDF paths. The actual dataset name is
# specified individually, as a meta dataset ('sec' by default) is used for
# validation and consistency checks (as FLASH files have holes everywhere)
# The eVMI and iTrace data sources are marked as packed, i.e. the indexer
# will pack their data into custom binary files for very efficient I/O.
gmd = flash.hdf('FL1/Photon Diagnostic/GMD/Pulse resolved energy',
                'energy BDA copy')
eVMI = flash.hdf('FL1/Experiment/BL1/CAMP/Pike Camera 2', 'image', packed=True)
iTrace = flash.hdf('FL1/Experiment/BL1/ADQ412 GHz ADC/CH00', 'TD')

# The data recorded by the FLASH machine DAQ has the same format, but slightly
# different paths. Such a DataRoot may therefore be created as a child of the
# FlashProposal above.
machine = FlashMachine(flash)
vls = machine.hdf('FL1/Photon Diagnostic/Wavelength/VLS online spectrometer',
                  'PCO.ROI.x')

# The pimms data uses its own root, as it also records different run IDs.
pimms = PimmsRoot(flash)
iDet = pimms.data()

# This call is optional, but here some settings of xts may be changed. The
# 'ahead-of-time' indexing strategy is currently the only implemented, but the
# interface is ready for 'just-in-time' indexing (which is much slower at runtime
# of course, but does not require an indexer to run before). In addition, the
# path of the index database file may be specified by a DataRoot, e.g. for the
# DataRoot to suggest a path. In this case it will suggest .../processed/)
xts.index_opts(strategy='aot', path=flash)
```

In addition to `FlashProposal`, `FlashMachine` (150 lines for indexing and I/O) and `PimmsRoot` (100 lines for indexing and I/O), there is support for EuropeanXFEL and LCLS. The toolkit is build on several layers of abstraction, so any manner of DAQ may be implemented. The most fundamental layer only expects a concept of "Get me data of source XXX for train ID YYYY". An `ArrayData` object may be used to simply wrap in-memory data and then use the map machinery in the same way.

In addition, the toolkit provides a number of useful mathematical functions in its `xts.math` subpackage.


## Quick reference

### _Module-level_

* `env(*args, **kwargs)` Load a path pointing to a configuration file (or search for .xtsenv.py if a directory) or add any data roots or sources via keywords arguments to the current environment.
* `alloc_array(shape: Union[int, Tuple[int]], dtype: numpy.dtype, per_worker: bool = False) -> numpy.ndarray` Allocate an array suitable for parallel execution. Depending on the parallelization settings, this might be backed by shared memory via mmap. `per_worker` prepends an axis as long as the number of workers. 
* `ordered` Custom slice implementation to allow indexing into `OrderedTrainSet` in non-ascending order, use like `numpy.s_`, i.e. `xts.ordered[:5]`.

### TrainSet


* Operators / Data model
  * `__iter__` Iterates over the list of train IDs.
  * `__len__() -> int` Returns the number of trains.
  * `__eq__(y: TrainSet) -> bool` Checks whether two sets refer to the same trains. 
  * `__and__(y: TrainSet) -> TrainSet` Returns the intersection of two sets.
  * `__or__(y: TrainSet) -> TrainSet` Returns the union of two sets.
  * `__getitem__(index) -> Union[int, TrainSet]` Returns singular train IDs when indexed by integers and sets by slices. Slices also support float argument for relative slicing.

* Utility methods
  * `split(num: int) -> List[TrainSet]` Split this set into a list of `num` sets.
  * `get_index_map() -> Dict[int, int]` Return a mapping of train ID to train position in this set, i.e. to index into a linear buffer. 
  * `peek_format(*ds: DataSource) -> List[Tuple], List[numpy.dtype]` Return a list of the shapes and dtypes for the specified data sources, obtained by the first data element of this set.
  * `peek_shape(*ds) -> List[Tuple]` Like `peek_format`, but only return the list of shapes.
  * `peek_dtype(*ds) -> List[numpy.dtype]` Like `peek_format`, but only return the list of dtypes.
  * `alloc_for_trains(*ds, shape=(,), dtype=numpy.float64, multiply_rows=1, **kwargs) -> numpy.ndarray` Shortcut for alloc_array to automatically choose the first axis corresponding to the number of trains in this set. If a single data source is passed, the shape and dtype is determined via peek_dtype. Additional keywords argument are passed to alloc_array
  * `alloc_for_reduce(ds, shape=None, dtype=None, **kwargs) -> numpy.ndarray` Shortcut for alloc_array to allocate buffers suitable for reduction operations. Unless specified, the shape and dtype is obtained via peek_format for a single data item and per_worker=True is passed to alloc_array.

* Map operations
  * `select_records(*ds) -> TrainSet` Return a new set which contains only those trains, where all specified data sources have records.
  * `map_trains(kernel, *ds)` Execute the given kernel on all trains on this set passing the specified data sources.
  * `select_trains(kernel, *ds) -> TrainSet` Return a new set for which the given kernel returned True.
  * `order_trains(kernel, *ds, pos_dtype=numpt.float64, pos_shape=(,)) -> OrderedTrainSet` Return an ordered set using the given kernel result to order the trains. `pos_dtype`/`pos_shape` must be compatible with the return value of the given kernel.
  * `load_trains(*ds) -> Tuple[numpy.ndarray]` Return the actual data for the specified data sources for all trains in this set. The format is chosen according to `peek_format`.
  * `average_trains(*ds) -> Tuple[numpy.ndarray]` Return the average for the specified data sources for all trains in this set. Ths format is chosen according to `peek_format`. Each data record is casted to a `numpy.float64`, added up and the result divided by the number of trains in this set.
  * `iterate_trains(*ds) -> Iterator[Tuple[int, ...]]` Return a generator to iterate over the actual data for the specified data sources. The generator returns a tuple of train ID and data. Not an map operation and not parallelized! 

## OrderedTrainSet
Special version of TrainSet, from which it inherits. Allows a custom order of train IDs (i.e. not in ascending order) for indexing purposes, only, as data access is always guaranteed to be in ascending order for I/O efficiency. To index into this set in its custom order, use the `ordered` slice implementation.

* `same_order_as(y: OrderedTrainSet) -> bool` Check whether another ordered set is in the same order (in addition to containing the same train IDs).
* `unorder() -> TrainSet` Convert the ordered set back into a simple set.

## Run
Construct a TrainSet via run IDs, custom identifiers to a particular data root. The set may be restricted by any number of run IDs and results in the intersection. The arguments may either be positional in order of data root definition or via keyword arguments using the data root symbol defined in a configuration file.