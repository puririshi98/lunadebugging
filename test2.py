import math
import random
import string
import unittest
import io
import unittest.mock as mock
import itertools
import warnings
import pickle
from copy import deepcopy
from itertools import repeat, product
from functools import reduce
from operator import mul
from collections import OrderedDict

import torch

# TODO: remove this global setting
# NN tests use double as the default dtype
torch.set_default_dtype(torch.double)

from torch._six import inf, nan
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import torch.nn.utils.parametrize as parametrize
import torch.nn.utils.prune as prune
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.nn import Parameter
from torch.nn.parameter import UninitializedParameter, UninitializedBuffer
from torch.nn.parallel._functions import Broadcast
from torch.testing import get_all_fp_dtypes
from torch.testing._internal.common_utils import freeze_rng_state, run_tests, TestCase, skipIfNoLapack, skipIfRocm, \
	TEST_NUMPY, TEST_SCIPY, TEST_WITH_ROCM, download_file, \
	get_function_arglist, load_tests, repeat_test_for_types, ALL_TENSORTYPES, \
	ALL_TENSORTYPES2, suppress_warnings, TemporaryFileName, TEST_WITH_UBSAN, IS_PPC
from torch.testing._internal.common_cuda import TEST_CUDA, TEST_MULTIGPU, TEST_CUDNN, TEST_CUDNN_VERSION
from torch.testing._internal.common_nn import NNTestCase, NewModuleTest, CriterionTest, \
	module_tests, criterion_tests, loss_reference_fns, \
	ctcloss_reference, new_module_tests
from torch.testing._internal.common_device_type import instantiate_device_type_tests, dtypes, \
	dtypesIfCUDA, precisionOverride, skipCUDAIfNoCudnn, skipCUDAIfCudnnVersionLessThan, onlyCUDA, onlyCPU, \
	skipCUDAIfRocm, skipCUDAIf, skipCUDAIfNotRocm, onlyOnCPUAndCUDA, \
	deviceCountAtLeast, largeTensorTest, expectedFailureMeta, skipMeta
from torch.nn import MultiheadAttention

from hypothesis import given
import torch.testing._internal.hypothesis_utils as hu
from torch.testing._internal.common_utils import _assertGradAndGradgradChecks, gradcheck, gradgradcheck, \
	GRADCHECK_NONDET_TOL
from torch.testing._internal.common_utils import dtype2prec_DONTUSE
from torch.testing._internal.common_cuda import tf32_on_and_off, tf32_is_not_fp32, tf32_off, tf32_on
from torch.types import _TensorOrTensors


AMPERE_OR_ROCM = TEST_WITH_ROCM or tf32_is_not_fp32()

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

if TEST_SCIPY:
	from scipy import stats
	import scipy.ndimage

if TEST_NUMPY:
	import numpy as np

DOUBLE_TENSORTYPES = [torch.double]


# WARNING: If you add a new top-level test case to this file, you MUST
# update test/run_test.py to list it, otherwise it will NOT be run in
class TestNNDeviceType(NNTestCase):


	@dtypes(torch.float, torch.double)
	@largeTensorTest(lambda self, device, dtype:
					 # Compute sum of the large tensor sizes:
					 # (im.numel() + small_image.numel() + small_image.grad.numel() +
					 #   large_view.grad.numel()) * sizeof(dtype)
					 32769 * (65536 + 3 * 65536 / 128) *
					 torch.tensor([], dtype=dtype).element_size())
	def test_grid_sample_large_index_2d(self, device, dtype):
		# Test 64-bit indexing with grid_sample (gh-41656)
		# Try accessing the corners, there should be no segfault
		coords = torch.tensor([[[-1., -1.],
								[+1., -1.]],

							   [[-1., +1.],
								[+1., +1.]]], device=device, dtype=dtype)
		coords = coords.expand(1, 2, 2, 2)
		im = torch.zeros([1, 1, 32769, 65536], device=device, dtype=dtype)

		# Compare sampling with large strides to the same op on a contiguous tensor
		coords = torch.rand(1, 4, 4, 2, device=device, dtype=dtype)
		large_view = im[..., 127::128]
		small_image = torch.rand_like(large_view)
		large_view[...] = small_image
		large_view.requires_grad, small_image.requires_grad = True, True
		self.assertTrue(
			sum(i * s for i, s in zip(large_view.size(), large_view.stride())) >= 2 ** 31,
			msg="View must use 64-bit indexing")
		for mode, padding_mode, align_corners in itertools.product(
				('nearest', 'bilinear', 'bicubic'), ('zeros', 'border', 'reflection'), (True, False)):
			a = F.grid_sample(
				small_image, coords, mode=mode,
				padding_mode=padding_mode, align_corners=align_corners)
			a.sum().backward()

			b = F.grid_sample(
				large_view, coords, mode=mode,
				padding_mode=padding_mode, align_corners=align_corners)
			b.sum().backward()

			self.assertEqual(a, b)
			self.assertEqual(small_image.grad, large_view.grad)

			small_image.grad.zero_()
			large_view.grad.zero_()

	@dtypes(torch.float, torch.double)
	@largeTensorTest(lambda self, device, dtype:
					 # Compute sum of the large tensor sizes:
					 # (im.numel() + small_image.numel() + small_image.grad.numel() +
					 #   large_view.grad.numel()) * sizeof(dtype)
					 2 * 32769 * (32768 + 3 * 32768 / 128) *
					 torch.tensor([], dtype=dtype).element_size())
	def test_grid_sample_large_index_3d(self, device, dtype):
		# Test 64-bit indexing with grid_sample (gh-41656)
		# Try accessing the corners, there should be no segfault
		coords = torch.full((1, 2, 2, 2, 3), 1., device=device, dtype=dtype)
		im = torch.zeros([1, 1, 2, 32769, 32768], device=device, dtype=dtype)

		result = F.grid_sample(im, coords, align_corners=False)
		self.assertEqual(result, torch.zeros((1, 1, 2, 2, 2), device=device, dtype=dtype))

		# Compare sampling with large strides to the same op on a contiguous tensor
		coords = torch.rand(1, 1, 4, 4, 3, device=device, dtype=dtype)
		large_view = im[..., 127::128]
		small_image = torch.rand_like(large_view)
		large_view[...] = small_image
		small_image.requires_grad, large_view.requires_grad = True, True
		self.assertTrue(
			sum(i * s for i, s in zip(large_view.size(), large_view.stride())) >= 2 ** 31,
			msg="View must use 64-bit indexing")
		for mode, padding_mode, align_corners in itertools.product(
				('nearest', 'bilinear'), ('zeros', 'border', 'reflection'), (True, False)):
			a = F.grid_sample(
				small_image, coords, mode=mode,
				padding_mode=padding_mode, align_corners=align_corners)
			a.sum().backward()

			b = F.grid_sample(
				large_view, coords, mode=mode,
				padding_mode=padding_mode, align_corners=align_corners)
			b.sum().backward()

			self.assertEqual(a, b)
			self.assertEqual(small_image.grad, large_view.grad)

			small_image.grad.zero_()
			large_view.grad.zero_()


instantiate_device_type_tests(TestNNDeviceType, globals())


if __name__ == '__main__':
	run_tests()
