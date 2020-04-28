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

	global __shmdebug__
	__shmdebug__ = flag


def get_debug():
  
	global __shmdebug__
	return __shmdebug__


def total_memory():

	with file('/proc/meminfo', 'r') as f:
		for line in f:
			words = line.split()
		if words[0].upper() == 'MEMTOTAL:':
			return int(words[1]) * 1024
	raise IOError('MemTotal unknown')


def cpu_count():

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
		self.counter.value = self.counter.value + 1
		self.event.set()


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
		e, r = self.result.get()
		self.slave.join()
		self.slave = None
		self.result = None
		if isinstance(e, Exception):
			raise SlaveException(e, r)
		return r

	
	
def MapReduceByThread(np=None):
	
	return MapReduce(backend=ThreadBackend, np=np)




class MapReduce(object):
	
	def __init__(self, backend=ProcessBackend, np=None):
		self.backend = backend
		if np is None:
			self.np = cpu_count()
		else:
			self.np = np

	def _main(self, pg, Q, R, sequence, realfunc):
		
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
				Q.cancel_join_thread()
				R.cancel_join_thread()
			pg.killall()
			pg.join()
			feeder.join()
			raise


def empty(shape, dtype='f8'):
	return anonymousmemmap(shape, dtype)

	
def full(shape, value, dtype='f8'):
	shared = empty(shape, dtype)
	shared[:] = value
	return shared

def copy(a):
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
	
	if ai['strides']:
		tp *= ai['strides'][-1]
	else:
		tp *= dtype.itemsize

	for i in numpy.asarray(ai['shape'])[::-1]:
		tp *= i

	ra = tp.from_address(ai['data'][0])
	buffer = numpy.ctypeslib.as_array(ra).ravel()

	shm = numpy.ndarray(buffer=buffer, dtype=dtype, 
			strides=ai['strides'], shape=ai['shape']).view(type=anonymousmemmap)
	return shm



class anonymousmemmap(numpy.memmap):

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

		return numpy.ndarray.__array_wrap__(self.view(numpy.ndarray), outarr, context)

	def __reduce__(self):
		return __unpickle__, (self.__array_interface__, self.dtype)


