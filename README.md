# Project Title

xts (eXtensible Toolkit for Shot analysis) provides a way to select arbitrary sets of (FEL) trains and then execute a custom function for each train with the data any number of detectors or other sources recorded for that train - without any need for manual I/O as well as automatic scaling across any number of cores. With a thin layer of facility-specific code written once to specify how to index and read the raw data, the location and format of the data is invisible and irrelevant during the actual data analysis.

## Getting Started

WIP!

## Quick guide
The fundamental object is TrainSet, which is an abstraction for an ordered list of train IDs. This object may be used to execute a kernel on this set or it may “select”ed into smaller TrainSets based upon arbitrary criteria (e.g. by a kernel returning bool).

```python
import xts

# Load the configuration file that defines the environment of this beamtime.
xts.load_rc(‘~/flash_chiral’)

# The run object initially contains all trains belonging to this run, while
# select_records(‘eVMI’) selects # only those where actual data for the passed
# data sources is present (in this case, the electron VMI). Run is a special
# subclass of TrainSet allowing addressing by run number, while
# select_records(*ds) returns just a TrainSet, which is a subset of the original # run. If this is not performed, the kernel may be called with None for some
# data sources (in particular for FLASH, where data is distributed randomly…)
run = xts.Run(flash=25367).select_records(‘eVMI’, ‘gmd’)

# alloc_array reserves memory and returns a numpy array for use in the kernels.
# Depending upon the parallelization configuration, this may be allocated in a
# shared memory region or even subdivided into further arrays for each worker
# (e.g. to parallelize a reduce operation).
evmi_data = xts.alloc_array((len(run), 1280, 960), np.float32, per_worker=False)

# Returns a dictionary mapping trainIds to buffer positions in the same order.
tid_map = run.get_index_map()

# The kernel acts on each train individually and is passed a workerId, trainId
# and the data sources requested
def read_kernel(wid, tid, eVMI_data, gmd_data):
    evmi_data[tid_map[tid]] = eVMI_data / gmd_data

# Perform the map operation.
run.map_trains(read_kernel, ‘eVMI’, ‘gmd’)

# evmi_data now contains the data. By default, it is parallelized via four
# worker processes. It the kernel releases the GIL itself, the method may also
# be changed to threads.
```

The configuration file contains the specific environment of a beamtime. It is defined by a collection of DataRoot objects and DataSource objects. A DataRoot defines the way data is organized by train ID, while the DataSource is then responsible for actually loading data for a specific train ID. In order to considerably speed up the reading process, data may be indexed (and packed into a highly efficient data format) transparently. This is implemented via subclasses of DataRoot/DataSource, which may instead by extended to gain this functionality. In particular for FLASH, where any read access to a file (even if only for a single train) requires around 1-2s of decompression, this can speed up the I/O by a factor of 50-100x. The configuration for the sample above is:

```python
import xts

from xts.flash import FlashProposal, FlashMachine
from xts.pimms import PimmsRoot

# Define the DataRoot for the data recorded by the FLASH user DAQ. If called
# within the GPFS path of this proposal, it may also be created by
# flash = FlashProposal.auto_detect()
flash = FlashProposal(year=2018, beamline=’bl1’, proposal=11004732)

# Define a number of data sources by HDF paths. The actual dataset name is
# specified individually, as a meta dataset (‘sec’ by default) is used for
# validation and consistency checks (as FLASH files have holes everywhere)
# The eVMI and iTrace data sources are marked as packed, i.e. the indexer
# will pack their data into custom binary files for very efficient I/O.
gmd = flash.hdf(‘FL1/Photon Diagnostic/GMD/Pulse resolved energy’,
                ‘energy BDA copy’)
eVMI = flash.hdf(‘FL1/Experiment/BL1/CAMP/Pike Camera 2’, ‘image’, packed=True)
iTrace = flash.hdf(‘FL1/Experiment/BL1/ADQ412 GHz ADC/CH00’, ‘TD’)

# The data recorded by the FLASH machine DAQ has the same format, but slightly
# different paths. Such a DataRoot may therefore be created as a child of the
# FlashProposal above.
machine = FlashMachine(flash)
vls = machine.hdf(‘FL1/Photon Diagnostic/Wavelength/VLS online spectrometer’,
                  ‘PCO.ROI.x’)

# The pimms data uses its own root, as it also records different run IDs.
pimms = PimmsRoot(flash)
iDet = pimms.data()

# This call is optional, but here some settings of xts may be changed. The
# ‘ahead-of-time’ indexing strategy is currently the only implemented, but the
# interface is ready for ‘just-in-time’ indexing (which is much slower at runtime
# of course, but does not require an indexer to run before). In addition, the
# path of the index database file may be specified by a DataRoot, e.g. for the
# DataRoot to suggest a path. In this case it will suggest .../processed/)
xts.index_opts(strategy=’aot’, path=flash)
```

In addition to FlashProposal, FlashMachine (150 lines for indexing and I/O) and PimmsRoot (100 lines for indexing and I/O), a corresponding XfelProposal implementation is almost done to apply the same functionality to EuXFEL beamtimes. The toolkit is build on several layers of abstraction, so any manner of DAQ may be implemented (e.g. psana). The most fundamental layer only expects a concept of “Get me data of source XXX for train ID YYYY”. An ArrayData object may be used to simply wrap in-memory data and then use the map machinery in the same way.

The remaining functionality of flashlib, that is radial/angular profiles, covariance mapping and auxiliary functions for quick detector readout will be implemented soon as utility functions using TrainSet objects.