#! /home/lyjslay/py3env/bin python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 * Ltd. All rights reserved.
#
#   File name   : shared_ram.py
#   Author      : lyjsly
#   Created date: 2020-03-22
#   Description : faster share numpy array between process
#
#================================================================
import os
import multiprocessing
import threading
try:
	import Queue as queue
except ImportError:
	import queue

from collections import deque
import traceback
import warnings
import gc
import threading
import heapq
import os

try:
	import cPickle as pickle
except ImportError:
	import pickle

import numpy
from multiprocessing import RawArray
import ctypes
import mmap


__shmdebug__ = False

def set_debug(flag):
	""" Set the debug mode.
		In debug mode (flag==True), the MapReduce pool will
		run the work function on the master thread / process.
		This ensures all exceptions can be properly inspected by 
		a debugger, e.g. pdb.

		Parameters
		----------
		flag : boolean
			   True for debug mode, False for production mode.
	"""
	global __shmdebug__
	__shmdebug__ = flag


def get_debug():
	""" Get the debug mode.
		Returns
		-------
		The debug mode. True if currently in debugging mode.
	"""    
	global __shmdebug__
	return __shmdebug__


def total_memory():
	""" Returns the the amount of memory available for use.

		The memory is obtained from MemTotal entry in /proc/meminfo.
		
		Notes
		-----
		This function is not very useful and not very portable. 

	"""
	with file('/proc/meminfo', 'r') as f:
		for line in f:
			words = line.split()
		if words[0].upper() == 'MEMTOTAL:':
			return int(words[1]) * 1024
	raise IOError('MemTotal unknown')


def cpu_count():
	""" Returns the default number of slave processes to be spawned.

		The default value is the number of physical cpu cores seen by python.
		:code:`OMP_NUM_THREADS` environment variable overrides it.

		On PBS/torque systems if OMP_NUM_THREADS is empty, we try to
		use the value of :code:`PBS_NUM_PPN` variable.

		Notes
		-----
		On some machines the physical number of cores does not equal
		the number of cpus shall be used. PSC Blacklight for example.

	"""
	num = os.getenv("OMP_NUM_THREADS")
	if num is None:
		num = os.getenv("PBS_NUM_PPN")
	try:
		return int(num)
	except:
		return multiprocessing.cpu_count()



class Ordered(object):
	def __init__(self, backend):
	  #  self.counter = lambda : None
		#multiprocessing.RawValue('l')
		self.event = backend.EventFactory()
		self.counter = multiprocessing.RawValue('l')
		self.tls = backend.StorageFactory()

	def reset(self):
		self.counter.value = 0
		self.event.set()

	def move(self, iter):
		self.tls.iter = iter

	def __enter__(self):
		while self.counter.value != self.tls.iter:
			self.event.wait() 
		self.event.clear()
		return self

	def __exit__(self, *args):
		# increase counter before releasing the value
		# so that the others waiting will see the new counter
		self.counter.value = self.counter.value + 1
		self.event.set()


class ThreadBackend:
	  QueueFactory = staticmethod(queue.Queue)
	  EventFactory = staticmethod(threading.Event)
	  LockFactory = staticmethod(threading.Lock)
	  StorageFactory = staticmethod(threading.local)
	  @staticmethod
	  def SlaveFactory(*args, **kwargs):
		slave = threading.Thread(*args, **kwargs)
		slave.daemon = True
		return slave

class ProcessBackend:
	  QueueFactory = staticmethod(multiprocessing.Queue)
	  EventFactory = staticmethod(multiprocessing.Event)
	  LockFactory = staticmethod(multiprocessing.Lock)

	  @staticmethod
	  def SlaveFactory(*args, **kwargs):
		slave = multiprocessing.Process(*args, **kwargs)
		slave.daemon = True
		return slave
	  @staticmethod
	  def StorageFactory():
		  return lambda:None

class background(object):
	""" Asyncrhonized function call via a background process.

		Parameters
		----------
		function : callable, the function to call
		*args   : positional arguments
		**kwargs : keyward arguments
	"""
	def __init__(self, function, *args, **kwargs):
			
		backend = kwargs.pop('backend', ProcessBackend)

		self.result = backend.QueueFactory(1)
		self.slave = backend.SlaveFactory(target=self._closure, 
				args=(function, args, kwargs, self.result))
		self.slave.start()

	def _closure(self, function, args, kwargs, result):
		try:
			rt = function(*args, **kwargs)
		except Exception as e:
			result.put((e, traceback.format_exc()))
		else:
			result.put((None, rt))

	def wait(self):
		""" Wait and join the child process. 
			The return value of the function call is returned.
			If any exception occurred it is wrapped and raised.
		"""
		e, r = self.result.get()
		self.slave.join()
		self.slave = None
		self.result = None
		if isinstance(e, Exception):
			raise SlaveException(e, r)
		return r

def MapReduceByThread(np=None):
	""" Creates a MapReduce object but with the Thread backend.

		The process backend is usually preferred.
	"""
	return MapReduce(backend=ThreadBackend, np=np)

class MapReduce(object):
	"""
		A pool of slave processes for a Map-Reduce operation

		Parameters
		----------
		backend : ProcessBackend or ThreadBackend
			ProcessBackend is preferred. ThreadBackend can be used in cases where
			processes creation is not allowed.

		np   : int or None
			Number of processes to use. Default (None) is from OMP_NUM_THREADS or
			the number of available cores on the computer. If np is 0, all operations
			are performed on the master process -- no child processes are created.

		Attributes
		----------
		np   : int
			Number of processes to use. (`omp_get_num_threads()`)

		local : object
			A namespace object that contains variables local to the worker
			thread / process. local is only accessible in the worker processes.

		local.rank : int
			The rank of the current worker. (`omp_get_thread_num()`)

		Notes
		-----
		Always wrap the call to :py:meth:`map` in a context manager ('with') block.
	"""
	def __init__(self, backend=ProcessBackend, np=None):
		self.backend = backend
		if np is None:
			self.np = cpu_count()
		else:
			self.np = np

	def _main(self, pg, Q, R, sequence, realfunc):
		# get and put will raise SlaveException
		# and terminate the process.
		# the exception is muted in ProcessGroup,
		# as it will only be dispatched from master.
		self.local = pg._tls
		try:
			while True:
				capsule = pg.get(Q)
				if capsule is None:
					return
				if len(capsule) == 1:
					i, = capsule
					work = sequence[i]
				else:
					i, work = capsule
				self.ordered.move(i)
				r = realfunc(work)
				pg.put(R, (i, r))
		except BaseException as e:
			if self.backend is ProcessBackend:
				# terminate the join threads of Queues to avoid deadlocks
				Q.cancel_join_thread()
				R.cancel_join_thread()
			raise
		self.local = None

	def __enter__(self):
		self.critical = self.backend.LockFactory()
		self.ordered = Ordered(self.backend)
		self.local = None # will be set during _main
		return self

	def __exit__(self, *args):
		self.ordered = None
		self.local = None
		pass

	def map(self, func, sequence, reduce=None, star=False, minlength=0):
		""" Map-reduce with multile processes.

			Apply func to each item on the sequence, in parallel. 
			As the results are collected, reduce is called on the result.
			The reduced result is returned as a list.
			
			Parameters
			----------
			func : callable
				The function to call. It must accept the same number of
				arguments as the length of an item in the sequence.

				.. warning::

					func is not supposed to use exceptions for flow control.
					In non-debug mode all exceptions will be wrapped into
					a :py:class:`SlaveException`.

			sequence : list or array_like
				The sequence of arguments to be applied to func.

			reduce : callable, optional
				Apply an reduction operation on the 
				return values of func. If func returns a tuple, they
				are treated as positional arguments of reduce.

			star : boolean
				if True, the items in sequence are treated as positional
				arguments of reduce.

			minlength: integer
				Minimal length of `sequence` to start parallel processing.
				if len(sequence) < minlength, fall back to sequential
				processing. This can be used to avoid the overhead of starting
				the worker processes when there is little work.
				
			Returns
			-------
			results : list
				The list of reduced results from the map operation, in
				the order of the arguments of sequence.
				
			Raises
			------
			SlaveException
				If any of the slave process encounters
				an exception. Inspect :py:attr:`SlaveException.reason` for the underlying exception.
		
		""" 
		def realreduce(r):
			if reduce:
				if isinstance(r, tuple):
					return reduce(*r)
				else:
					return reduce(r)
			return r

		def realfunc(i):
			if star: return func(*i)
			else: return func(i)

		if len(sequence) <= 0 or self.np == 0 or get_debug():
			# Do this in serial
			self.local = lambda : None
			self.local.rank = 0

			rt = [realreduce(realfunc(i)) for i in sequence]

			self.local = None
			return rt

		# never use more than len(sequence) processes
		np = min([self.np, len(sequence)])

		Q = self.backend.QueueFactory(64)
		R = self.backend.QueueFactory(64)
		self.ordered.reset()

		pg = ProcessGroup(main=self._main, np=np,
				backend=self.backend,
				args=(Q, R, sequence, realfunc))

		pg.start()

		L = []
		N = []
		def feeder(pg, Q, N):
			#   will fail silently if any error occurs.
			j = 0
			try:
				for i, work in enumerate(sequence):
					if not hasattr(sequence, '__getitem__'):
						pg.put(Q, (i, work))
					else:
						pg.put(Q, (i, ))
					j = j + 1
				N.append(j)

				for i in range(np):
					pg.put(Q, None)
			except StopProcessGroup:
				return
			finally:
				pass
		feeder = threading.Thread(None, feeder, args=(pg, Q, N))
		feeder.start()

		# we run fetcher on main thread to catch exceptions
		# raised by reduce 
		count = 0
		try:
			while True:
				try:
					capsule = pg.get(R)
				except queue.Empty:
					continue
				except StopProcessGroup:
					raise pg.get_exception()
				capsule = capsule[0], realreduce(capsule[1])
				heapq.heappush(L, capsule)
				count = count + 1
				if len(N) > 0 and count == N[0]: 
					# if finished feeding see if all
					# results have been obtained
					break
			rt = []

			while len(L) > 0:
				rt.append(heapq.heappop(L)[1])
			pg.join()
			feeder.join()
			assert N[0] == len(rt)
			return rt
		except BaseException as e:
			if self.backend is ProcessBackend:
				# terminate the join threads of Queues to avoid deadlocks.
				Q.cancel_join_thread()
				R.cancel_join_thread()
			pg.killall()
			pg.join()
			feeder.join()
			raise


def empty_like(array, dtype=None):
	""" 
	Create a shared memory array from the shape of array.
	"""
	array = numpy.asarray(array)
	if dtype is None: 
		dtype = array.dtype
	return anonymousmemmap(array.shape, dtype)

def empty(shape, dtype='f8'):
	""" 
	Create an empty shared memory array.
	"""
	return anonymousmemmap(shape, dtype)

def full_like(array, value, dtype=None):
	""" 
	Create a shared memory array with the same shape and type as a given array, filled with 'value'.
	"""
	shared = empty_like(array, dtype)
	shared[:] = value
	return shared
	
def full(shape, value, dtype='f8'):
	""" 
	Create a shared memory array of given shape and type, filled with `value`.
	"""
	shared = empty(shape, dtype)
	shared[:] = value
	return shared

def copy(a):
	""" 
	Copy an array to the shared memory. 

		Notes
		-----
		copy is not always necessary because the private memory is always copy-on-write.

		Use :code:`a = copy(a)` to immediately dereference the old 'a' on private memory
	"""
	shared = anonymousmemmap(a.shape, dtype=a.dtype)
	shared[:] = a[:]
	return shared

def fromiter(iter, dtype, count=None):
	return copy(numpy.fromiter(iter, dtype, count))

try:
	# numpy >= 1.16
	_unpickle_ctypes_type = numpy.ctypeslib.as_ctypes_type(numpy.dtype('|u1'))
except:
	# older version numpy < 1.16
	_unpickle_ctypes_type = numpy.ctypeslib._typecodes['|u1']

def __unpickle__(ai, dtype):
	dtype = numpy.dtype(dtype)
	tp = _unpickle_ctypes_type * 1

	# if there are strides, use strides, otherwise the stride is the itemsize of dtype
	if ai['strides']:
		tp *= ai['strides'][-1]
	else:
		tp *= dtype.itemsize

	for i in numpy.asarray(ai['shape'])[::-1]:
		tp *= i

	# grab a flat char array at the sharemem address, with length at least contain ai required
	ra = tp.from_address(ai['data'][0])
	buffer = numpy.ctypeslib.as_array(ra).ravel()
	# view it as what it should look like
	shm = numpy.ndarray(buffer=buffer, dtype=dtype, 
			strides=ai['strides'], shape=ai['shape']).view(type=anonymousmemmap)
	return shm

class anonymousmemmap(numpy.memmap):
	""" 
	Arrays allocated on shared memory. 
	The array is stored in an anonymous memory map that is shared between child-processes.
	"""
	def __new__(subtype, shape, dtype=numpy.uint8, order='C'):

		descr = numpy.dtype(dtype)
		_dbytes = descr.itemsize

		shape = numpy.atleast_1d(shape)
		size = 1
		for k in shape:
			size *= k

		bytes = int(size*_dbytes)

		if bytes > 0:
			mm = mmap.mmap(-1, bytes)
		else:
			mm = numpy.empty(0, dtype=descr)
		self = numpy.ndarray.__new__(subtype, shape, dtype=descr, buffer=mm, order=order)
		self._mmap = mm
		return self
		
	def __array_wrap__(self, outarr, context=None):
	# after ufunc this won't be on shm!
		return numpy.ndarray.__array_wrap__(self.view(numpy.ndarray), outarr, context)

	def __reduce__(self):
		return __unpickle__, (self.__array_interface__, self.dtype)


