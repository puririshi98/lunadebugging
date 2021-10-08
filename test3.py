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
    def _test_dropout(self, cls, device, input, memory_format=torch.contiguous_format):
        p = 0.2
        input = input.to(device).fill_(1 - p)

        module = cls(p)
        input_var = input.clone(memory_format=memory_format).requires_grad_()
        output = module(input_var)
        self.assertTrue(output.is_contiguous(memory_format=memory_format))
        self.assertLess(abs(output.data.mean() - (1 - p)), 0.05)
        output.backward(input)
        self.assertTrue(input_var.grad.is_contiguous(memory_format=memory_format))
        self.assertLess(abs(input_var.grad.data.mean() - (1 - p)), 0.05)

        module = cls(p, True)
        input_var = input.clone(memory_format=memory_format).requires_grad_()
        output = module(input_var + 0)
        self.assertTrue(output.is_contiguous(memory_format=memory_format))
        self.assertLess(abs(output.data.mean() - (1 - p)), 0.05)
        output.backward(input)
        self.assertTrue(input_var.grad.is_contiguous(memory_format=memory_format))
        self.assertLess(abs(input_var.grad.data.mean() - (1 - p)), 0.05)

        # check eval mode doesn't change anything
        for inplace in [True, False]:
            module = cls(p, inplace).eval()
            self.assertEqual(input, module(input))

        # Check that these don't raise errors
        module.__repr__()
        str(module)

    def _test_dropout_discontiguous(self, cls, device, memory_format=torch.contiguous_format):
        # In this test, we verify that dropout preserves the layout and data for different memory formats.
        # We check whether, we get same values for the output of dropout, when the probability
        # of dropout is 0 or very close to 0.
        # Reference: https://github.com/pytorch/pytorch/issues/47176
        close_to_zero_p = 1e-10  # Should be almost zero but not zero, as for p=0 different path is taken
        for p in [0, close_to_zero_p]:
            inp = torch.ones(2, 3, 3, 3, device=device)
            inp_discontiguous = torch.empty(2, 3, 3, 6, device=device, memory_format=memory_format)[..., ::2]
            inp_discontiguous.copy_(inp)
            mod = cls(p=p)
            out = mod(inp_discontiguous)
            if p != 0:  # Zero will keep strides as is based on input.
                # When prob == 0, input stride (54, 18, 6, 2) -> output stride (54, 18, 6, 2)
                # When prob != 0, input stride (54, 18, 6, 2) -> output stride (27, 9, 3, 1)
                self.assertTrue(out.is_contiguous(memory_format=memory_format))
            self.assertEqual(inp_discontiguous, out)

    def _test_dropout_stride_mean_preserve(self, cls, device):
        def invert_perm(p):
            d = {x: i for i, x in enumerate(p)}
            return (d[0], d[1], d[2], d[3])

        inp = torch.ones(2, 3, 4, 5, device=device)
        shifts = [(0, 0), (1, 0), (0, 1), (1, 1)]
        for perm in itertools.permutations((0, 1, 2, 3), r=4):
            for shift in shifts:
                for p in [1e-10, 0.3, 0.5, 0.7]:
                    mod = cls(p=p)
                    permuted_inp = inp.permute(perm).contiguous().permute(invert_perm(perm))
                    permuted_inp = permuted_inp[shift[0]:, shift[1]:, :, :]
                    out = mod(permuted_inp)

                    self.assertTrue(out.permute(perm).is_contiguous())
                    self.assertEqual(inp.mean(), out.mean(), rtol=0.5, atol=0.5)
                    if p == 1e-10:
                        self.assertEqual(permuted_inp, out)
                    else:
                        self.assertNotEqual(permuted_inp, out)

    def _test_InstanceNorm_general(self, cls, input, device, dtype=torch.float):
        # default case track_running_stats=False
        b, c = input.size(0), input.size(1)
        input_var = input.to(device=device, dtype=dtype).requires_grad_()

        IN = cls(c, eps=0).to(device, dtype)

        output = IN(input_var)
        out_reshaped = output.view(b * c, -1)

        mean = out_reshaped.mean(1)
        var = out_reshaped.var(1, unbiased=False)

        self.assertEqual(torch.abs(mean.data).mean(), 0, atol=1e-5, rtol=0)
        self.assertEqual(torch.abs(var.data).mean(), 1, atol=1e-5, rtol=0)

        # check that eval mode doesn't change behavior
        grad_out = torch.randn_like(output)
        res1 = output.data.clone()
        output.backward(grad_out)
        grad1 = input_var.grad.data.clone()

        IN.eval()
        output = IN(input_var)
        input_var.grad = None
        output.backward(grad_out)
        res2 = output.data
        grad2 = input_var.grad.data
        self.assertEqual(res1, res2)
        self.assertEqual(grad1, grad2)

        # If track_running_stats=True and momentum=1, running_mean/var should be
        # equal to mean/var of the input (with unbias correction)
        IN = cls(c, momentum=1, eps=0, track_running_stats=True).to(device, dtype)

        output = IN(input_var)

        input_reshaped = input_var.transpose(1, 0).reshape(c, -1)
        mean = input_reshaped.mean(1)

        input_reshaped = input_var.transpose(1, 0).reshape(c, b, -1)
        var = input_reshaped.var(2, unbiased=True)[:, :]

        self.assertEqual(torch.abs(mean.data - IN.running_mean).mean(), 0, atol=1e-5, rtol=0)
        self.assertEqual(torch.abs(var.data.mean(1) - IN.running_var).mean(), 0, atol=1e-5, rtol=0)

        # in eval mode, adding X * std to a channel in input should make the
        # corresponding channel in output have mean X
        IN.eval()
        delta = IN.running_var.sqrt() * torch.arange(c, device=device, dtype=dtype)
        delta = delta.view(-1, *[1 for _ in range(2, input.dim())])
        output = IN(input_var + delta)
        self.assertEqual(output.transpose(0, 1).reshape(c, -1).mean(1), torch.arange(c, dtype=dtype))

    def _test_InstanceNorm_cuda_half(self, cls, input, device):
        # THNN
        input = input.to(device=device, dtype=torch.half).random_(1, 10).requires_grad_(True)
        m = cls(input.size(1), affine=True, track_running_stats=True).to(device, torch.half)
        thnn_output = m(input)
        thnn_output.sum().backward()
        thnn_input_grad = input.grad.data.clone()
        self.assertEqualTypeString(thnn_output, input)
        # cuDNN
        if TEST_CUDNN:
            input.grad = None
            m = m.float()
            cudnn_output = m(input)
            cudnn_output.sum().backward()
            cudnn_input_grad = input.grad.data.clone()
            self.assertEqualTypeString(cudnn_output, input)
            self.assertEqual(cudnn_output, thnn_output, atol=1e-4, rtol=0)
            self.assertEqual(cudnn_input_grad, thnn_input_grad, atol=1e-3, rtol=0)

    def _test_LayerNorm_general(self, device, dtype=torch.float):
        for i in range(2, 6):
            shape = torch.randint(3, 6, (i,), dtype=torch.long).tolist()
            x = torch.empty(*shape, device=device, dtype=dtype).uniform_(0, 10)
            normalized_ndim = random.randint(1, i - 1)  # inclusive
            normalized_shape = shape[-normalized_ndim:]
            unnormalized_shape = shape[:-normalized_ndim]

            # test that LN normalizes to mean 0 and stddev 1
            ln = nn.LayerNorm(normalized_shape, eps=0).to(device, dtype)
            ln.weight.data.fill_(1)
            ln.bias.data.fill_(0)
            output = ln(x)
            out_reshaped = output.view(*(unnormalized_shape + [-1]))
            mean = out_reshaped.mean(-1)
            var = out_reshaped.var(-1, unbiased=False)

            delta = 1e-1 if dtype == torch.bfloat16 else 1e-5
            self.assertEqual(torch.abs(mean.data).mean(), 0, atol=delta, rtol=0)
            self.assertEqual(torch.abs(var.data).mean(), 1, atol=delta, rtol=0)

            # test that LN applies weight and bias correctly
            scale, bias = torch.empty(2).uniform_(0.2, 2).tolist()
            ln.weight.data.fill_(scale)
            ln.bias.data.fill_(bias)
            output = ln(x)
            out_reshaped = output.view(*(unnormalized_shape + [-1]))
            mean = out_reshaped.mean(-1)
            var = out_reshaped.var(-1, unbiased=False)
            self.assertEqual(torch.abs(mean.data).mean(), bias, atol=delta, rtol=0)
            self.assertEqual(torch.abs(var.data).mean(), scale ** 2, atol=delta, rtol=0)

        bad_norm_shape_input_shape = {
            (): (),
            (2, 3): (3,),
            (2,): (1, 2, 3),
            (10,): (2, 3),
            10: (2, 3),
        }
        for norm_shape, input_shape in bad_norm_shape_input_shape.items():
            ln = nn.LayerNorm(norm_shape)
            input = torch.empty(input_shape, device=device, dtype=dtype).uniform_(0, 10)
            self.assertRaises(RuntimeError, lambda: ln(input))

    def _test_LayerNorm_cuda_half(self, device):
        input = torch.empty(2, 3, 3, 2, device=device, dtype=torch.half).random_(1, 10).requires_grad_(True)
        m = nn.LayerNorm([3, 2]).to(device, torch.half)
        output = m(input)
        output.sum().backward()
        self.assertEqualTypeString(output, input)

    def _test_GroupNorm_general(self, device, dtype=torch.float):
        good_shape_g = {
            (1, 2, 3, 4): 2,
            (2, 3, 10): 3,
            (3, 1, 1, 1, 2): 1,
            (2, 6, 4, 2, 2): 3,
            (1, 256, 1, 1): 32,
        }
        for shape_g, grad in product(good_shape_g.items(), [True, False]):
            shape, g = shape_g
            x = torch.empty(*shape, device=device, dtype=dtype).uniform_(0, 10)
            x.requires_grad_(grad)
            b = shape[0]
            c = shape[1]

            # test that GN normalizes to mean 0 and stddev 1
            gn = nn.GroupNorm(g, c, eps=0).to(device, dtype)
            gn.weight.data.fill_(1)
            gn.bias.data.fill_(0)
            output = gn(x)
            out_reshaped = output.view(b, g, -1)
            mean = out_reshaped.mean(-1)
            var = out_reshaped.var(-1, unbiased=False)
            # TODO: fix numerical issue. See #44863
            self.assertEqual(torch.abs(mean).mean(), 0, atol=1e-3, rtol=1e-3)
            self.assertEqual(torch.abs(var).mean(), 1, atol=1e-3, rtol=1e-3)

            output.backward(torch.randn_like(output))
            if output.is_cuda:
                torch.cuda.synchronize()

            # test that GN applies weight and bias correctly
            scale = torch.empty(c, device=device, dtype=dtype).uniform_(0.2, 2)
            bias = torch.empty(c, device=device, dtype=dtype).uniform_(0.2, 2)
            gn.weight.data.copy_(scale)
            gn.bias.data.copy_(bias)
            output = gn(x)
            out_reshaped = output.view(b, c, -1)
            out_normed = (out_reshaped - bias.view(c, 1)) / scale.view(c, 1)
            out_normed_reshaped = out_normed.view(b, g, -1)
            mean = out_normed_reshaped.mean(-1)
            var = out_normed_reshaped.var(-1, unbiased=False)
            # TODO: fix numerical issue. See #44863
            self.assertEqual(torch.abs(mean).mean(), 0, atol=1e-3, rtol=1e-3)
            self.assertEqual(torch.abs(var).mean(), 1, atol=1e-3, rtol=1e-3)

        bad_shape_g = {
            (1, 2, 3, 4): 3,
            (2, 3, 10): 2,
            (3, 1, 1, 1, 2): 10,
            (2, 6, 4, 2, 2): 4,
        }
        for shape, g in bad_shape_g.items():
            gn = nn.GroupNorm(g, shape[1])
            input = torch.empty(*shape, device=device, dtype=dtype).uniform_(0, 10)
            self.assertRaises(RuntimeError, lambda: gn(input))

    def _test_GroupNorm_cuda_half(self):
        input = torch.zeros(2, 4, 3, 2, requires_grad=True).cuda().half().random_(1, 10)
        m = nn.GroupNorm(2, 4).to("cuda", torch.half)
        output = m(input)
        output.sum().backward()
        self.assertEqualTypeString(output, input)

    def _test_module_empty_input(self, module, inp, check_size=True):
        inp.requires_grad_(True)
        out = module(inp)
        gO = torch.rand_like(out)
        out.backward(gO)
        if check_size:
            self.assertEqual(out.size(), inp.size())
        for p in module.parameters():
            if p.requires_grad:
                self.assertEqual(p.grad, torch.zeros_like(p.grad))
        self.assertEqual(inp.grad, torch.zeros_like(inp))

    @unittest.skipIf((not TEST_NUMPY) or (not TEST_SCIPY) or (scipy.__version__ < '1.0.0'),
                     "Scipy v1.0 and/or numpy not found")
    @tf32_on_and_off()
    def test_affine_2d_rotate0(self, device):
        # scipy before 1.0.0 do not support homogeneous coordinate
        # scipy.ndimage.affine_transform, so we need to skip.
        input_size = [1, 1, 3, 3]
        input_ary = np.array(np.random.random(input_size), dtype=np.float32)
        output_size = [1, 1, 5, 5]
        angle_rad = 0.

        transform_tensor, transform_ary, offset = \
            _buildEquivalentAffineTransforms2d(device, input_size, output_size, angle_rad)

        scipy_ary = torch.from_numpy(scipy.ndimage.affine_transform(
            input_ary[0, 0],
            transform_ary,
            offset=offset,
            output_shape=output_size[2:],
            order=1,
            mode='nearest',
            prefilter=False))

        affine_tensor = torch.nn.functional.affine_grid(
            transform_tensor,
            torch.Size(output_size),
            align_corners=True
        )

        gridsample_ary = torch.nn.functional.grid_sample(
            torch.tensor(input_ary, device=device).to(device),
            affine_tensor,
            padding_mode='border',
            align_corners=True
        ).to('cpu')

        self.assertEqual(scipy_ary.mean(), gridsample_ary.mean())
        self.assertEqual(scipy_ary, gridsample_ary.reshape_as(scipy_ary))

    @unittest.skipIf((not TEST_NUMPY) or (not TEST_SCIPY) or (scipy.__version__ < '1.0.0'),
                     "Scipy v1.0 and/or numpy not found")
    @tf32_on_and_off(0.001)
    def test_affine_2d_rotate90(self, device):
        # scipy before 1.0.0 do not support homogeneous coordinate
        # scipy.ndimage.affine_transform, so we need to skip.
        for input_size2dsq, output_size2dsq in \
                itertools.product(input_size2dsq_(), output_size2dsq_()):
            input_size = input_size2dsq
            input_ary = np.array(np.random.random(input_size), dtype=np.float32)
            output_size = output_size2dsq
            angle_rad = 0.25 * math.pi * 2

            transform_tensor, transform_ary, offset = \
                _buildEquivalentAffineTransforms2d(device, input_size, output_size, angle_rad)

            scipy_ary = torch.from_numpy(scipy.ndimage.affine_transform(
                input_ary[0, 0],
                transform_ary,
                offset=offset,
                output_shape=output_size[2:],
                order=1,
                mode='nearest',
                prefilter=True))

            if input_size2dsq == output_size2dsq:
                self.assertEqual(scipy_ary.mean(), input_ary.mean())
            self.assertEqual(scipy_ary[0, 0], input_ary[0, 0, 0, -1])
            self.assertEqual(scipy_ary[0, -1], input_ary[0, 0, -1, -1])
            self.assertEqual(scipy_ary[-1, -1], input_ary[0, 0, -1, 0])
            self.assertEqual(scipy_ary[-1, 0], input_ary[0, 0, 0, 0])

            affine_tensor = torch.nn.functional.affine_grid(
                transform_tensor,
                torch.Size(output_size),
                align_corners=True
            )

            gridsample_ary = torch.nn.functional.grid_sample(
                torch.tensor(input_ary, device=device).to(device),
                affine_tensor,
                padding_mode='border',
                align_corners=True
            ).to('cpu')

            self.assertEqual(scipy_ary.mean(), gridsample_ary.mean())
            self.assertEqual(scipy_ary, gridsample_ary.reshape_as(scipy_ary))

    @unittest.skipIf((not TEST_NUMPY) or (not TEST_SCIPY) or (scipy.__version__ < '1.0.0'),
                     "Scipy v1.0 and/or numpy not found")
    @tf32_on_and_off(0.005)
    def test_affine_2d_rotate45(self, device):
        # scipy before 1.0.0 do not support homogeneous coordinate
        # scipy.ndimage.affine_transform, so we need to skip.
        input_size = [1, 1, 3, 3]
        input_ary = np.array(np.zeros(input_size), dtype=np.float32)
        input_ary[0, 0, 0, :] = 0.5
        input_ary[0, 0, 2, 2] = 1.0
        output_size = [1, 1, 3, 3]
        angle_rad = 0.125 * math.pi * 2

        transform_tensor, transform_ary, offset = \
            _buildEquivalentAffineTransforms2d(device, input_size, output_size, angle_rad)

        scipy_ary = torch.from_numpy(scipy.ndimage.affine_transform(
            input_ary[0, 0],
            transform_ary,
            offset=offset,
            output_shape=output_size[2:],
            order=1,
            mode='nearest',
            prefilter=False))

        affine_tensor = torch.nn.functional.affine_grid(
            transform_tensor,
            torch.Size(output_size),
            align_corners=True
        )

        gridsample_ary = torch.nn.functional.grid_sample(
            torch.tensor(input_ary, device=device).to(device),
            affine_tensor,
            padding_mode='border',
            align_corners=True
        ).to('cpu')

        self.assertEqual(scipy_ary, gridsample_ary.reshape_as(scipy_ary))

    @unittest.skipIf((not TEST_NUMPY) or (not TEST_SCIPY) or (scipy.__version__ < '1.0.0'),
                     "Scipy v1.0 and/or numpy not found")
    @tf32_on_and_off(0.005)
    def test_affine_2d_rotateRandom(self, device):
        # scipy before 1.0.0 do not support homogeneous coordinate
        # scipy.ndimage.affine_transform, so we need to skip.
        for angle_rad, input_size2d, output_size2d in \
                itertools.product(angle_rad_(), input_size2d_(), output_size2d_()):

            input_size = input_size2d
            input_ary = np.array(np.random.random(input_size), dtype=np.float32).round(3)
            output_size = output_size2d

            input_ary[0, 0, 0, 0] = 2
            input_ary[0, 0, 0, -1] = 4
            input_ary[0, 0, -1, 0] = 6
            input_ary[0, 0, -1, -1] = 8

            transform_tensor, transform_ary, grid_ary = \
                _buildEquivalentAffineTransforms2d(device, input_size, output_size, angle_rad)

            scipy_ary = torch.from_numpy(scipy.ndimage.affine_transform(
                input_ary[0, 0],
                transform_ary,
                output_shape=output_size[2:],
                order=1,
                mode='nearest',
                prefilter=False))

            affine_tensor = torch.nn.functional.affine_grid(
                transform_tensor,
                torch.Size(output_size),
                align_corners=True
            )

            gridsample_ary = torch.nn.functional.grid_sample(
                torch.tensor(input_ary, device=device).to(device),
                affine_tensor,
                padding_mode='border',
                align_corners=True
            ).to('cpu')

            affine_tensor = affine_tensor.to('cpu')

            for r in range(affine_tensor.size(1)):
                for c in range(affine_tensor.size(2)):
                    grid_out = np.dot(grid_ary, [r, c, 1])
                    self.assertEqual(affine_tensor[0, r, c], grid_out[:2], exact_dtype=False)

            self.assertEqual(scipy_ary, gridsample_ary.reshape_as(scipy_ary))

    @unittest.skipIf((not TEST_NUMPY) or (not TEST_SCIPY) or (scipy.__version__ < '1.0.0'),
                     "Scipy v1.0 and/or numpy not found")
    @tf32_on_and_off(0.005)
    def test_affine_3d_rotateRandom(self, device):
        # scipy before 1.0.0 do not support homogeneous coordinate
        # scipy.ndimage.affine_transform, so we need to skip.
        for angle_rad, axis_vector, input_size3d, output_size3d in \
                itertools.product(angle_rad_(), axis_vector_(), input_size3d_(), output_size3d_()):
            input_size = input_size3d
            input_ary = np.array(np.random.random(input_size), dtype=np.float32)
            output_size = output_size3d

            input_ary[0, 0, 0, 0, 0] = 2
            input_ary[0, 0, 0, 0, -1] = 3
            input_ary[0, 0, 0, -1, 0] = 4
            input_ary[0, 0, 0, -1, -1] = 5
            input_ary[0, 0, -1, 0, 0] = 6
            input_ary[0, 0, -1, 0, -1] = 7
            input_ary[0, 0, -1, -1, 0] = 8
            input_ary[0, 0, -1, -1, -1] = 9

            transform_tensor, transform_ary, grid_ary = \
                _buildEquivalentAffineTransforms3d(device, input_size, output_size, angle_rad, axis_vector)

            scipy_ary = torch.from_numpy(scipy.ndimage.affine_transform(
                input_ary[0, 0],
                transform_ary,
                output_shape=output_size[2:],
                order=1,
                mode='nearest',
                prefilter=False))

            affine_tensor = torch.nn.functional.affine_grid(
                transform_tensor,
                torch.Size(output_size),
                align_corners=True
            )

            gridsample_ary = torch.nn.functional.grid_sample(
                torch.tensor(input_ary, device=device).to(device),
                affine_tensor,
                padding_mode='border',
                align_corners=True
            ).to('cpu')

            affine_tensor = affine_tensor.to('cpu')

            for i in range(affine_tensor.size(1)):
                for r in range(affine_tensor.size(2)):
                    for c in range(affine_tensor.size(3)):
                        grid_out = np.dot(grid_ary, [i, r, c, 1])
                        self.assertEqual(affine_tensor[0, i, r, c], grid_out[:3], exact_dtype=False)

            self.assertEqual(scipy_ary, gridsample_ary.reshape_as(scipy_ary))

    def test_conv1d_same_padding(self, device):
        # Test padding='same' outputs the correct shape
        test_args = [
            # in_size
            range(50, 55),
            # kernel_size
            [1, 2, 3, 8],
            # dilation
            range(1, 4),
            # stride
            [1],
        ]
        for in_size, k_size, dilation, stride in itertools.product(*test_args):
            x = torch.rand(1, 1, in_size, device=device)
            y = torch.rand(1, 1, k_size, device=device)
            z = F.conv1d(x, y, padding='same', dilation=dilation, stride=stride)
            self.assertEqual(z.size(2), int(math.ceil(in_size / stride)))

        # Compare F.conv1d padding='same' output against manual padding
        # Without strides/dilation
        x = torch.rand(1, 1, 12, device=device)
        y = torch.rand(1, 1, 3, device=device)
        expect = F.conv1d(x, y, padding=1)
        actual = F.conv1d(x, y, padding='same')
        self.assertEqual(expect, actual)

        # With dilation
        x = torch.rand(1, 1, 12, device=device)
        y = torch.rand(1, 1, 4, device=device)
        expect = F.conv1d(x, y, padding=3, dilation=2)
        actual = F.conv1d(x, y, padding='same', dilation=2)
        self.assertEqual(expect, actual)

        # Dilation with asymmetric padding
        expect = F.conv1d(x, y, padding=5, dilation=3)[..., 1:]
        actual = F.conv1d(x, y, padding='same', dilation=3)
        self.assertEqual(expect, actual)


    def test_conv2d_same_padding(self, device):
        # Compare F.conv2d padding='same' output against manual padding
        # Without strides/dilation
        x = torch.rand(1, 1, 10, 11, device=device)
        y = torch.rand(1, 1, 4, 5, device=device)
        expect = F.conv2d(x, y, padding=(2, 2))[..., 1:, :]
        actual = F.conv2d(x, y, padding='same')
        self.assertEqual(expect, actual)

        # With dilation
        y = torch.rand(1, 1, 3, 4, device=device)
        expect = F.conv2d(x, y, padding=(2, 3), dilation=2)
        actual = F.conv2d(x, y, padding='same', dilation=2)
        self.assertEqual(expect, actual)

        # Dilation with asymmetric padding
        y = torch.rand(1, 1, 4, 4, device=device)
        expect = F.conv2d(x, y, padding=5, dilation=3)[..., 1:, 1:]
        actual = F.conv2d(x, y, padding='same', dilation=3)
        self.assertEqual(expect, actual)

    def test_conv3d_same_padding(self, device):
        # Compare F.conv3d padding='same' output against manual padding
        # Without strides/dilation
        x = torch.rand(1, 1, 10, 11, 12, device=device)
        y = torch.rand(1, 1, 1, 2, 5, device=device)
        expect = F.conv3d(x, y, padding=(0, 1, 2))[..., :, 1:, :]
        actual = F.conv3d(x, y, padding='same')
        self.assertEqual(expect, actual)

        # With dilation
        expect = F.conv3d(x, y, padding=(0, 1, 4), dilation=2)
        actual = F.conv3d(x, y, padding='same', dilation=2)
        self.assertEqual(expect, actual)

        # Dilation with asymmetric padding
        y = torch.rand(1, 1, 4, 4, 4, device=device)
        expect = F.conv3d(x, y, padding=5, dilation=3)[..., 1:, 1:, 1:]
        actual = F.conv3d(x, y, padding='same', dilation=3)
        self.assertEqual(expect, actual)

    def test_conv1d_valid_padding(self, device):
        # Test F.conv1d padding='valid' is the same as no padding
        x = torch.rand(1, 1, 10, device=device)
        y = torch.rand(1, 1, 4, device=device)
        expect = F.conv1d(x, y)
        actual = F.conv1d(x, y, padding='valid')
        self.assertEqual(expect, actual)

    def test_conv2d_valid_padding(self, device):
        # Test F.conv2d padding='valid' is the same as no padding
        x = torch.rand(1, 1, 1, 10, device=device)
        y = torch.rand(1, 1, 1, 4, device=device)
        expect = F.conv2d(x, y)
        actual = F.conv2d(x, y, padding='valid')
        self.assertEqual(expect, actual)

    def test_conv3d_valid_padding(self, device):
        # Test F.conv3d padding='valid' is the same as no padding
        x = torch.rand(1, 1, 1, 1, 10, device=device)
        y = torch.rand(1, 1, 1, 1, 4, device=device)
        expect = F.conv3d(x, y)
        actual = F.conv3d(x, y, padding='valid')
        self.assertEqual(expect, actual)

    def test_conv1d_same_padding_backward(self, device):
        # Test F.conv1d gradients work with padding='same'
        x = torch.rand(1, 1, 12, device=device, requires_grad=True)
        y = torch.rand(1, 1, 4, device=device, requires_grad=True)

        # Symmetric padding
        z = F.conv1d(x, y, padding=3, dilation=2)
        z.sum().backward()
        gx_expect, gy_expect = x.grad, y.grad
        x.grad, y.grad = None, None

        z = F.conv1d(x, y, padding='same', dilation=2)
        z.sum().backward()
        self.assertEqual(gx_expect, x.grad)
        self.assertEqual(gy_expect, y.grad)
        x.grad, y.grad = None, None

        # Asymmetric padding
        z = F.conv1d(x, y, padding=2)[..., 1:]
        z.sum().backward()
        gx_expect, gy_expect = x.grad, y.grad
        x.grad, y.grad = None, None

        z = F.conv1d(x, y, padding='same')
        z.sum().backward()
        self.assertEqual(gx_expect, x.grad)
        self.assertEqual(gy_expect, y.grad)

    def test_conv2d_same_padding_backward(self, device):
        # Test F.conv2d gradients work with padding='same'
        x = torch.rand(1, 1, 10, 11, device=device, requires_grad=True)
        y = torch.rand(1, 1, 4, 5, device=device, requires_grad=True)

        # Symmetric padding
        z = F.conv2d(x, y, padding=(3, 4), dilation=2)
        z.sum().backward()
        gx_expect, gy_expect = x.grad, y.grad
        x.grad, y.grad = None, None

        z = F.conv2d(x, y, padding='same', dilation=2)
        z.sum().backward()
        self.assertEqual(gx_expect, x.grad)
        self.assertEqual(gy_expect, y.grad)
        x.grad, y.grad = None, None

        # Asymmetric padding
        y = torch.rand(1, 1, 4, 4, device=device, requires_grad=True)
        z = F.conv2d(x, y, padding=2)[..., 1:, 1:]
        z.sum().backward()
        gx_expect, gy_expect = x.grad, y.grad
        x.grad, y.grad = None, None

        z = F.conv1d(x, y, padding='same')
        z.sum().backward()
        self.assertEqual(gx_expect, x.grad)
        self.assertEqual(gy_expect, y.grad)

    def test_conv3d_same_padding_backward(self, device):
        # Test F.conv3d gradients work with padding='same'
        x = torch.rand(1, 1, 1, 11, 12, device=device, requires_grad=True)
        y = torch.rand(1, 1, 1, 2, 5, device=device, requires_grad=True)

        # Symmetric padding
        z = F.conv3d(x, y, padding=(0, 1, 4), dilation=2)
        z.sum().backward()
        gx_expect, gy_expect = x.grad, y.grad
        x.grad, y.grad = None, None

        z = F.conv3d(x, y, padding='same', dilation=2)
        z.sum().backward()
        self.assertEqual(gx_expect, x.grad)
        self.assertEqual(gy_expect, y.grad)
        x.grad, y.grad = None, None

        # Asymmetric padding
        y = torch.rand(1, 1, 1, 4, 4, device=device, requires_grad=True)
        z = F.conv3d(x, y, padding=2)[..., 1:, 1:]
        z.sum().backward()
        gx_expect, gy_expect = x.grad, y.grad
        x.grad, y.grad = None, None

        z = F.conv3d(x, y, padding='same')
        z.sum().backward()
        self.assertEqual(gx_expect, x.grad)
        self.assertEqual(gy_expect, y.grad)

    def test_conv1d_valid_padding_backward(self, device):
        # Test F.conv1d gradients work with padding='valid'
        x = torch.rand(1, 1, 10, device=device, requires_grad=True)
        y = torch.rand(1, 1, 4, device=device, requires_grad=True)
        F.conv1d(x, y, padding=0).sum().backward()
        gx_expect, gy_expect = x.grad, y.grad
        x.grad, y.grad = None, None

        F.conv1d(x, y, padding='valid').sum().backward()
        gx_actual, gy_actual = x.grad, y.grad
        self.assertEqual(gx_expect, gx_actual)
        self.assertEqual(gy_expect, gy_actual)

    def test_conv2d_valid_padding_backward(self, device):
        # Test F.conv2d gradients work with padding='valid'
        x = torch.rand(1, 1, 1, 10, device=device, requires_grad=True)
        y = torch.rand(1, 1, 1, 4, device=device, requires_grad=True)
        F.conv2d(x, y, padding=0).sum().backward()
        gx_expect, gy_expect = x.grad, y.grad
        x.grad, y.grad = None, None

        F.conv2d(x, y, padding='valid').sum().backward()
        gx_actual, gy_actual = x.grad, y.grad
        self.assertEqual(gx_expect, gx_actual)
        self.assertEqual(gy_expect, gy_actual)

    def test_conv3d_valid_padding_backward(self, device):
        # Test F.conv3d gradients work with padding='valid'
        x = torch.rand(1, 1, 1, 1, 10, device=device, requires_grad=True)
        y = torch.rand(1, 1, 1, 1, 4, device=device, requires_grad=True)
        F.conv3d(x, y, padding=0).sum().backward()
        gx_expect, gy_expect = x.grad, y.grad
        x.grad, y.grad = None, None

        F.conv3d(x, y, padding='valid').sum().backward()
        gx_actual, gy_actual = x.grad, y.grad
        self.assertEqual(gx_expect, gx_actual)
        self.assertEqual(gy_expect, gy_actual)

    def test_Dropout(self, device):
        input = torch.empty(1000)
        self._test_dropout(nn.Dropout, device, input)

        self._test_dropout_discontiguous(nn.Dropout, device)
        self._test_dropout_discontiguous(nn.Dropout, device, memory_format=torch.channels_last)

        self._test_dropout_stride_mean_preserve(nn.Dropout, device)

        if self.device_type == 'cuda':
            input = input.bfloat16()
            self._test_dropout(nn.Dropout, device, input)

    def test_Dropout2d(self, device):
        b = random.randint(1, 5)
        w = random.randint(1, 5)
        h = random.randint(1, 5)
        num_features = 1000
        input = torch.empty(num_features, b, w, h)
        self._test_dropout(nn.Dropout2d, device, input)
        self._test_dropout(nn.Dropout2d, device, input, memory_format=torch.channels_last)

        self._test_dropout_discontiguous(nn.Dropout2d, device)
        self._test_dropout_discontiguous(nn.Dropout2d, device, memory_format=torch.channels_last)

    def test_Dropout3d(self, device):
        b = random.randint(1, 5)
        w = random.randint(1, 5)
        h = random.randint(1, 5)
        d = random.randint(1, 2)
        num_features = 1000
        input = torch.empty(num_features, b, d, w, h)
        self._test_dropout(nn.Dropout3d, device, input)

        self._test_dropout_discontiguous(nn.Dropout3d, device)
        self._test_dropout_discontiguous(nn.Dropout3d, device, memory_format=torch.channels_last)

    def test_InstanceNorm1d_general(self, device):
        b = random.randint(3, 5)
        c = random.randint(3, 5)
        d = random.randint(8, 10)

        input = torch.rand(b, c, d)
        self._test_InstanceNorm_general(nn.InstanceNorm1d, input, device)

        if self.device_type == 'cuda':
            self._test_InstanceNorm_cuda_half(nn.InstanceNorm1d, input, device)

    def test_InstanceNorm2d_general(self, device):
        b = random.randint(3, 5)
        c = random.randint(3, 5)
        w = random.randint(3, 6)
        h = random.randint(6, 8)

        input = torch.rand(b, c, h, w)
        self._test_InstanceNorm_general(nn.InstanceNorm2d, input, device)

        if self.device_type == 'cuda':
            self._test_InstanceNorm_cuda_half(nn.InstanceNorm2d, input, device)

    def test_InstanceNorm3d_general(self, device):
        b = random.randint(3, 5)
        c = random.randint(3, 5)
        w = random.randint(2, 5)
        h = random.randint(2, 5)
        d = random.randint(2, 5)

        input = torch.rand(b, c, h, w, d)
        self._test_InstanceNorm_general(nn.InstanceNorm3d, input, device)

        if self.device_type == 'cuda':
            self._test_InstanceNorm_cuda_half(nn.InstanceNorm3d, input, device)

    def test_instancenorm_raises_error_if_less_than_one_value_per_channel(self, device):
        x = torch.rand(10)[None, :, None]
        with self.assertRaises(ValueError):
            torch.nn.InstanceNorm1d(10)(x).to(device)

    def test_instancenorm_raises_error_for_single_spatial_element_during_training(self, device):
        BATCH_SIZE = 10
        NUM_CHANNELS = 3
        norms = [torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d]
        for i, norm in enumerate(norms):
            m = norm(NUM_CHANNELS, track_running_stats=True)
            m.to(device)

            # Create an appropriately-sized input with a single spatial element.
            input = torch.randn(BATCH_SIZE, NUM_CHANNELS, *[1 for _ in range(i + 1)],
                                device=device)
            with self.assertRaises(ValueError):
                m(input)

            # Single spatial element should be fine in eval.
            m.eval()
            m(input)

    def test_LayerNorm_general(self, device):
        self._test_LayerNorm_general(device)

        if self.device_type == 'cuda' or self.device_type == 'cpu':
            self._test_LayerNorm_general(device, dtype=torch.bfloat16)

        if self.device_type == 'cuda':
            self._test_LayerNorm_cuda_half(device)

    @onlyOnCPUAndCUDA
    def test_LayerNorm_numeric(self, device):
        def layer_norm_ref(X, gamma, beta, normalized_shape, eps):
            feature_size = np.prod(normalized_shape)
            X_view = X.view(-1, feature_size)
            mean = X_view.mean(dim=-1, keepdim=True)
            var = X_view.var(dim=-1, unbiased=False, keepdim=True)
            Y = (X_view - mean) / torch.sqrt(var + eps)
            Y = Y * gamma.view(-1) + beta.view(-1)
            return Y.view(*X.size())

        normalized_shape = [256, 256, 144]
        layer_norm = nn.LayerNorm(normalized_shape).float().to(device)
        X = torch.rand(2, *normalized_shape, dtype=torch.float32,
                       device=device)

        Y = layer_norm(X)
        Y_ref = layer_norm_ref(X, layer_norm.weight.data, layer_norm.bias.data,
                               normalized_shape, layer_norm.eps)
        self.assertEqual(Y, Y_ref, rtol=0, atol=1e-5)

        if self.device_type == 'cuda':
            layer_norm.cpu()
            Y_cpu = layer_norm(X.cpu())
            self.assertEqual(Y_cpu, Y, rtol=0, atol=1e-5)

    @onlyOnCPUAndCUDA
    def test_GroupNorm_general(self, device):
        self._test_GroupNorm_general(device)

        if self.device_type == 'cuda':
            self._test_GroupNorm_cuda_half()

    def test_GroupNorm_raises_error_if_one_value_per_group(self, device):
        x = torch.rand(10)[None, :, None]
        with self.assertRaises(ValueError):
            torch.nn.GroupNorm(10, 10)(x).to(device)

    def test_GroupNorm_empty(self, device):
        mod = torch.nn.GroupNorm(2, 4).to(device)
        inp = torch.randn(0, 4, 2, 2, device=device)
        self._test_module_empty_input(mod, inp)
        if self.device_type == 'cuda' and self.has_cudnn():
            with torch.backends.cudnn.flags(enabled=False):
                self._test_module_empty_input(mod, inp)

    @onlyOnCPUAndCUDA
    def test_GroupNorm_numeric(self, device):
        def group_norm_ref(X, gamma, beta, groups, channels, eps):
            batch_size = X.size()[0]
            X_view = X.view(batch_size, groups, -1)
            mean = X_view.mean(dim=-1, keepdim=True)
            var = X_view.var(dim=-1, unbiased=False, keepdim=True)
            Y = ((X_view - mean) / torch.sqrt(var + eps)).view(
                batch_size, channels, -1)
            Y = Y * gamma.view(channels, 1) + beta.view(channels, 1)
            return Y.view(*X.size())

        batch_size = 1
        groups = 2
        channels = 8
        group_norm = nn.GroupNorm(groups, channels).float().to(device)
        X = torch.rand(batch_size, channels, 256, 256, 72,
                       dtype=torch.float32, device=device)

        Y = group_norm(X)
        Y_ref = group_norm_ref(
            X, group_norm.weight.data, group_norm.bias.data, groups,
            channels, group_norm.eps)
        self.assertEqual(Y, Y_ref, rtol=0, atol=1e-5)

        if self.device_type == 'cuda':
            group_norm.cpu()
            Y_cpu = group_norm(X.cpu())
            self.assertEqual(Y_cpu, Y, rtol=0, atol=1e-5)

    @onlyOnCPUAndCUDA
    @dtypes(torch.float64, torch.complex128)
    def test_pad(self, device, dtype):
        inputs = torch.randn(1, 3, 4, 4, device=device, dtype=dtype, requires_grad=True)
        _assertGradAndGradgradChecks(self, lambda x: F.pad(x, (1, 1, 1, 1)), (inputs,),
                                     nondet_tol=GRADCHECK_NONDET_TOL)
        _assertGradAndGradgradChecks(self, lambda x: F.pad(x, (-1, 1, -2, 1)), (inputs,),
                                     nondet_tol=GRADCHECK_NONDET_TOL)
        _assertGradAndGradgradChecks(self, lambda x: F.pad(x, (-1, 1, -2, 1), value=2), (inputs,),
                                     nondet_tol=GRADCHECK_NONDET_TOL)
        self.assertTrue(gradcheck(lambda x: F.pad(x, (-1, 1, -2, 1), mode='replicate'), (inputs,),
                                  nondet_tol=GRADCHECK_NONDET_TOL))
        self.assertTrue(gradcheck(lambda x: F.pad(x, (-1, 1, -2, 1), mode='reflect'), (inputs,),
                                  nondet_tol=GRADCHECK_NONDET_TOL))
        self.assertTrue(gradcheck(lambda x: F.pad(x, (-1, 1, -2, 1), mode='circular'), (inputs,),
                                  nondet_tol=GRADCHECK_NONDET_TOL))

        inputs = torch.randn(1, 2, 3, 4, 4, device=device, dtype=dtype, requires_grad=True)
        self.assertTrue(gradcheck(lambda x: F.pad(x, (1, 1, 1, 1, 1, 1), mode='replicate'), (inputs,),
                                  nondet_tol=GRADCHECK_NONDET_TOL))

        # Assert assertion errors are raised for invalid circular padding values
        inputs = torch.randn(1, 1, 4, device=device, dtype=dtype, requires_grad=True)
        # Should raise error when trying to wrap around more than once
        self.assertRaises(AssertionError, lambda: F.pad(inputs, (5, 4), mode='circular'))
        self.assertRaises(AssertionError, lambda: F.pad(inputs, (3, 6), mode='circular'))
        # Should raise error when negative padding results in negative output shape
        self.assertRaises(AssertionError, lambda: F.pad(inputs, (-3, -2), mode='circular'))

        # assert that relfection padding errors when pad >= input size
        expected_err_msg = r"Padding size should be less than the corresponding input dimension"
        inputs = torch.randn(1, 1, 2, 3, device=device, dtype=dtype)
        self.assertRaisesRegex(RuntimeError, expected_err_msg,
                               lambda: F.pad(inputs, (1, 1, 3, 0), mode='reflect'))
        inputs = torch.randn(1, 1, 2, device=device, dtype=dtype)
        self.assertRaisesRegex(RuntimeError, expected_err_msg,
                               lambda: F.pad(inputs, (2, 1), mode='reflect'))

        inputs = torch.rand(1, 3, 4, 4, device=device, dtype=dtype)
        # assert that pad doesn't return a view into the input tensor
        for mode in 'constant', 'reflect', 'replicate', 'circular':
            out = F.pad(inputs, (0, 0, 0, 0), mode=mode)
            out.fill_(4)
            self.assertTrue(torch.all(torch.abs(inputs) < 2))

            out = F.pad(inputs, (0, 0, -1, -1), mode=mode)
            out.fill_(4)
            self.assertTrue(torch.all(torch.abs(inputs) < 2))

    @onlyOnCPUAndCUDA
    @dtypes(torch.float64, torch.complex128)
    def test_ReplicationPad_empty(self, device, dtype):
        for mod, inp in [
                (torch.nn.ReplicationPad1d(3), torch.randn(0, 3, 10, device=device, dtype=dtype)),
                (torch.nn.ReplicationPad2d(3), torch.randn(0, 3, 10, 10, device=device, dtype=dtype)),
                (torch.nn.ReplicationPad3d(3), torch.randn(0, 3, 10, 10, 10, device=device, dtype=dtype))]:
            self._test_module_empty_input(mod, inp, check_size=False)

        with self.assertRaisesRegex(NotImplementedError, 'Only 3D'):
            mod = torch.nn.ReplicationPad1d(2)
            inp = torch.randn(3, 10, device=device, dtype=dtype)
            mod(inp)

        with self.assertRaisesRegex(RuntimeError, 'Expected 2D or 3D'):
            mod = torch.nn.ReplicationPad1d(2)
            inp = torch.randn(3, 0, 10, device=device, dtype=dtype)
            mod(inp)

        with self.assertRaisesRegex(RuntimeError, 'Expected 3D or 4D'):
            mod = torch.nn.ReplicationPad2d((2, 2, 2, 2))
            inp = torch.randn(43, 0, 10, 10, device=device, dtype=dtype)
            mod(inp)

        with self.assertRaisesRegex(RuntimeError, 'Expected 4D or 5D'):
            mod = torch.nn.ReplicationPad3d((2, 2, 2, 2, 2, 2))
            inp = torch.randn(3, 0, 10, 10, 10, device=device, dtype=dtype)
            mod(inp)

    def test_ReplicationPad1d_large(self, device):
        shapes = ([2, 65736, 4], [65736, 2, 4])
        pl, pr = 3, 4
        for shape in shapes:
            x = torch.randn(shape, device=device, requires_grad=True)
            model = torch.nn.ReplicationPad1d((pl, pr))

            # forward
            out = model(x)
            self.assertEqual(out[:, :, pl : -pr], x)

            left_padding = out[:, :, : pl]
            self.assertEqual(left_padding, x[:, :, :1].expand_as(left_padding))
            right_padding = out[:, :, -pr :]
            self.assertEqual(right_padding, x[:, :, -1:].expand_as(right_padding))

            # backward
            g = torch.randn_like(out)
            out.backward(g)
            self.assertEqual(x.grad[:, :, 1 : -1], g[:, :, pl + 1 : -pr - 1])

            self.assertEqual(x.grad[:, :, 0], g[:, :, : pl + 1].sum(-1))
            self.assertEqual(x.grad[:, :, -1], g[:, :, -pr - 1:].sum(-1))

    def test_ReplicationPad2d_large(self, device):
        shapes = ([2, 65736, 4, 4], [65736, 2, 4, 4])
        pl, pr, pt, pb = 3, 4, 5, 6
        for shape in shapes:
            x = torch.randn(shape, device=device, requires_grad=True)
            model = torch.nn.ReplicationPad2d((pl, pr, pt, pb))

            # forward center, edge
            out = model(x)
            self.assertEqual(out[:, :, pt : -pb, pl : -pr], x)

            left_padding = out[:, :, pt : -pb, : pl]
            self.assertEqual(left_padding, x[:, :, :, :1].expand_as(left_padding))
            right_padding = out[:, :, pt : -pb, -pr :]
            self.assertEqual(right_padding, x[:, :, :, -1:].expand_as(right_padding))
            top_padding = out[:, :, : pt, pl : -pr]
            self.assertEqual(top_padding, x[:, :, :1, :].expand_as(top_padding))
            bottom_padding = out[:, :, -pb : , pl : -pr]
            self.assertEqual(bottom_padding, x[:, :, -1:, :].expand_as(bottom_padding))

            # forward corner
            tl_padding = out[:, :, : pt + 1, : pl + 1]
            self.assertEqual(tl_padding, x[:, :, :1, :1].expand_as(tl_padding))
            tr_padding = out[:, :, : pt + 1, -pr - 1:]
            self.assertEqual(tr_padding, x[:, :, :1, -1:].expand_as(tr_padding))
            bl_padding = out[:, :, -pb - 1:, : pl + 1]
            self.assertEqual(bl_padding, x[:, :, -1:, :1].expand_as(bl_padding))
            br_padding = out[:, :, -pb - 1:, -pr - 1:]
            self.assertEqual(br_padding, x[:, :, -1:, -1:].expand_as(br_padding))

            # backward center, edge
            g = torch.randn_like(out)
            out.backward(g)
            self.assertEqual(x.grad[:, :, 1:-1, 1:-1], g[:, :, pt + 1 : -pb - 1, pl + 1 : -pr - 1])

            self.assertEqual(x.grad[:, :, 1:-1, 0], g[:, :, pt + 1 : -pb - 1, : pl + 1].sum(-1))
            self.assertEqual(x.grad[:, :, 1:-1, -1], g[:, :, pt + 1 : -pb - 1, -pr - 1 :].sum(-1))
            self.assertEqual(x.grad[:, :, 0, 1:-1], g[:, :, : pt + 1, pl + 1 : -pr - 1].sum(-2))
            self.assertEqual(x.grad[:, :, -1, 1:-1], g[:, :, -pb - 1 :, pl + 1 : -pr - 1].sum(-2))

            # backward corner
            self.assertEqual(x.grad[:, :, 0, 0], g[:, :, : pt + 1, : pl + 1].sum((-2, -1)))
            self.assertEqual(x.grad[:, :, 0, -1], g[:, :, : pt + 1, -pr - 1 :].sum((-2, -1)))
            self.assertEqual(x.grad[:, :, -1, 0], g[:, :, -pb - 1 :, : pl + 1].sum((-2, -1)))
            self.assertEqual(x.grad[:, :, -1, -1], g[:, :, -pb - 1 :, -pr - 1 :].sum((-2, -1)))

    @largeTensorTest("6GB")
    def test_ReplicationPad3d_large(self, device):
        shapes = ([1, 65736, 2, 2, 2], [65736, 1, 2, 2, 2])
        pl, pr, pt, pbt, pf, pbk = 3, 4, 5, 6, 7, 8

        for shape in shapes:
            x = torch.randn(shape, device=device, requires_grad=True)
            model = torch.nn.ReplicationPad3d((pl, pr, pt, pbt, pf, pbk))

            # forward center
            out = model(x)
            self.assertEqual(out[:, :, pf : -pbk, pt : -pbt, pl : -pr], x)

            # backward center
            g = torch.randn_like(out)
            out.backward(g)
            self.assertEqual(x.grad[:, :, 1:-1, 1:-1, 1:-1], g[:, :, pf + 1 : -pbk - 1, pt + 1 : -pbt - 1, pl + 1 : -pr - 1])

    @onlyOnCPUAndCUDA
    @dtypes(torch.float32, torch.complex64)
    def test_ReflectionPad_empty(self, device, dtype):
        for mod, inp in [
                (torch.nn.ReflectionPad1d(2), torch.randn(0, 3, 10, device=device, dtype=dtype)),
                (torch.nn.ReflectionPad2d(2), torch.randn(0, 3, 10, 10, device=device, dtype=dtype)),
                (torch.nn.ReflectionPad3d(3), torch.randn(0, 3, 10, 10, 10, device=device, dtype=dtype))]:
            self._test_module_empty_input(mod, inp, check_size=False)

        with self.assertRaisesRegex(RuntimeError, '2D or 3D'):
            mod = torch.nn.ReflectionPad1d(2)
            inp = torch.randn(3, 0, 10, device=device, dtype=dtype)
            mod(inp)

        with self.assertRaisesRegex(RuntimeError, '3D or 4D'):
            mod = torch.nn.ReflectionPad2d(2)
            inp = torch.randn(3, 0, 10, 10, device=device, dtype=dtype)
            mod(inp)

        with self.assertRaisesRegex(RuntimeError, '4D or 5D'):
            mod = torch.nn.ReflectionPad3d(3)
            inp = torch.randn(3, 0, 10, 10, 10, device=device, dtype=dtype)
            mod(inp)

    @onlyCUDA   # Test if CPU and GPU results match
    def test_ReflectionPad2d_large(self, device):
        shapes = ([2, 65736, 6, 6], [65736, 2, 6, 6])
        pad = (1, 2, 3, 4)
        for shape in shapes:
            x = torch.randn(shape, device=device, requires_grad=True)
            ref_x = x.detach().cpu().requires_grad_()

            out = F.pad(x, pad, mode='reflect')
            ref_out = F.pad(ref_x, pad, mode='reflect')

            self.assertEqual(out, ref_out)

            g = torch.randn_like(out)
            ref_g = g.cpu()

            out.backward(g)
            ref_out.backward(ref_g)

            self.assertEqual(x.grad, ref_x.grad)

    @onlyCUDA   # Test if CPU and GPU results match
    def test_ReflectionPad3d_large(self, device):
        shapes = ([2, 1000, 7, 7, 7], [1000, 2, 7, 7, 7])
        pad = (1, 2, 3, 4, 5, 6)
        for shape in shapes:
            x = torch.randn(shape, device=device, requires_grad=True)
            ref_x = x.detach().cpu().requires_grad_()

            out = F.pad(x, pad, mode='reflect')
            ref_out = F.pad(ref_x, pad, mode='reflect')

            self.assertEqual(out, ref_out)

            g = torch.randn_like(out)
            ref_g = g.cpu()

            out.backward(g)
            ref_out.backward(ref_g)

            self.assertEqual(x.grad, ref_x.grad)

    @onlyOnCPUAndCUDA
    @dtypes(torch.float, torch.double)
    def test_MarginLoss_empty(self, device, dtype):
        for mod, x, y in [
                (torch.nn.MultiMarginLoss().to(device),
                 torch.randn(0, 10, requires_grad=True, device=device, dtype=dtype),
                 torch.ones(0, device=device).type(torch.long)),
                (torch.nn.MultiLabelMarginLoss().to(device),
                 torch.randn(0, 10, requires_grad=True, device=device, dtype=dtype),
                 torch.ones(0, 10, device=device).type(torch.long))]:

            out = mod(x, y)
            out.sum().backward()

            self.assertEqual(x, torch.zeros_like(x))
            self.assertEqual(x.grad, torch.zeros_like(x))

            with self.assertRaisesRegex(RuntimeError, 'Expected'):
                x = torch.randn(0, requires_grad=True, device=device, dtype=dtype)
                y = torch.ones(10, device=device).type(torch.long)
                mod(x, y)

            with self.assertRaisesRegex(RuntimeError, 'Expected'):
                x = torch.randn(10, 0, requires_grad=True, device=device, dtype=dtype)
                y = torch.ones(10, 0, device=device).type(torch.long)
                mod(x, y)


    @onlyOnCPUAndCUDA
    def test_Unfold_empty(self, device):
        inp = torch.randn(0, 3, 3, 4, device=device)
        unfold = torch.nn.Unfold(kernel_size=(2, 3)).to(device)
        self._test_module_empty_input(unfold, inp, check_size=False)

        with self.assertRaisesRegex(RuntimeError, 'Expected 3D or 4D'):
            inp = torch.randn(3, 0, 3, 4, device=device)
            unfold = torch.nn.Unfold(kernel_size=(2, 3)).to(device)
            unfold(inp)

    @onlyCUDA
    @dtypes(torch.float, torch.double)
    @tf32_on_and_off(0.005)
    def test_rnn_fused(self, device, dtype):

        def copy_rnn(rnn1, rnn2):
            for x_layer, y_layer in zip(rnn1.all_weights, rnn2.all_weights):
                for x, y in zip(x_layer, y_layer):
                    x.data.copy_(y.data)

        def check_rnn_grads(rnn1, rnn2):
            for x_layer, y_layer in zip(rnn1.all_weights, rnn2.all_weights):
                for x, y in zip(x_layer, y_layer):
                    self.assertEqual(x.grad, y.grad, atol=5e-5, rtol=0)

        input_size = 10
        hidden_size = 6
        num_layers = 2
        seq_length = 7
        batch = 6
        input_val = torch.randn(seq_length, batch, input_size, dtype=dtype)
        grad_output = torch.randn(seq_length, batch, hidden_size, dtype=dtype)
        hx_val = torch.randn(num_layers, batch, hidden_size, dtype=dtype)
        grad_hy = torch.randn(num_layers, batch, hidden_size, dtype=dtype)
        with torch.backends.cudnn.flags(enabled=False, allow_tf32=None):
            for module in (nn.GRU, nn.LSTM):
                for bias in (True, False):
                    rnn = module(input_size, hidden_size, num_layers, bias=bias).to(dtype)
                    rnn_device = module(input_size, hidden_size, num_layers, bias=bias).to(device, dtype)
                    copy_rnn(rnn, rnn_device)

                    is_lstm = isinstance(rnn, nn.LSTM)
                    if is_lstm:
                        hx = (hx_val.clone().requires_grad_(True),
                              hx_val.clone().add(1).requires_grad_(True))
                        hx_device = (hx_val.clone().to(device).requires_grad_(True),
                                     hx_val.clone().to(device).add(1).requires_grad_(True))
                    else:
                        hx = hx_val.clone().requires_grad_(True)
                        hx_device = hx_val.clone().to(device).requires_grad_(True)

                    inp = input_val.clone().requires_grad_(True)
                    inp_cu = input_val.clone().to(device).requires_grad_(True)
                    output1, hy1 = rnn(inp, hx)
                    output2, hy2 = rnn_device(inp_cu, hx_device)
                    if is_lstm:
                        torch.autograd.backward(
                            [output1, hy1[0], hy1[1]], [grad_output, grad_hy, grad_hy + 1]
                        )
                        torch.autograd.backward(
                            [output2, hy2[0], hy2[1]],
                            [grad_output.to(device), grad_hy.to(device), (grad_hy + 1).to(device)]
                        )
                    else:
                        torch.autograd.backward([output1, hy1], [grad_output, grad_hy])
                        torch.autograd.backward([output2, hy2], [grad_output.to(device), grad_hy.to(device)])

                    self.assertEqual(output1, output2)
                    self.assertEqual(hy1, hy2)

                    check_rnn_grads(rnn, rnn_device)
                    self.assertEqual(inp.grad, inp_cu.grad)
                    if is_lstm:
                        self.assertEqual(hx[0].grad, hx_device[0].grad)
                        self.assertEqual(hx[1].grad, hx_device[1].grad)
                    else:
                        self.assertEqual(hx.grad, hx_device.grad)

    def test_BatchNorm_empty(self, device):
        mod = torch.nn.BatchNorm2d(3).to(device)
        inp = torch.randn(0, 3, 2, 2, device=device)
        self._test_module_empty_input(mod, inp)
        if self.device_type == 'cuda' and self.has_cudnn():
            with torch.backends.cudnn.flags(enabled=False):
                self._test_module_empty_input(mod, inp)

        self.assertEqual(mod.running_mean, torch.tensor([0., 0, 0], device=device))
        self.assertEqual(mod.running_var, torch.tensor([1., 1, 1], device=device))
        self.assertEqual(mod.weight.grad, torch.tensor([0., 0, 0], device=device))
        self.assertEqual(mod.bias.grad, torch.tensor([0., 0, 0], device=device))

    def test_group_conv_empty(self, device):
        mod = torch.nn.Conv2d(4, 4, stride=2, kernel_size=3, padding=1, groups=4).to(device)
        inp = torch.randn(0, 4, 4, 4, device=device)
        self._test_module_empty_input(mod, inp, check_size=False)
        if self.device_type == 'cuda' and self.has_cudnn():
            with torch.backends.cudnn.flags(enabled=False):
                self._test_module_empty_input(mod, inp, check_size=False)

    def test_group_convTranspose_empty(self, device):
        mod = torch.nn.ConvTranspose2d(4, 4, stride=2, kernel_size=3, padding=1, groups=4).to(device)
        inp = torch.randn(0, 4, 4, 4, device=device)
        self._test_module_empty_input(mod, inp, check_size=False)
        if self.device_type == 'cuda' and self.has_cudnn():
            with torch.backends.cudnn.flags(enabled=False):
                self._test_module_empty_input(mod, inp, check_size=False)

    def test_convTranspose_empty(self, device):
        mod = torch.nn.ConvTranspose2d(4, 4, stride=2, kernel_size=3, padding=1).to(device)
        inp = torch.randn(0, 4, 4, 4, device=device)
        self._test_module_empty_input(mod, inp, check_size=False)
        if self.device_type == 'cuda' and self.has_cudnn():
            with torch.backends.cudnn.flags(enabled=False):
                self._test_module_empty_input(mod, inp, check_size=False)

    @onlyOnCPUAndCUDA
    def test_AvgPool2d_empty(self, device):
        avgpool = torch.nn.AvgPool2d(3, stride=2).to(device)
        inp = torch.randn(0, 16, 20, 32, device=device)
        self._test_module_empty_input(avgpool, inp, check_size=False)

        clast_inp = torch.randn(0, 16, 20, 32, device=device).contiguous(memory_format=torch.channels_last)
        self._test_module_empty_input(avgpool, clast_inp, check_size=False)

        # test with empty non-batch input
        with self.assertRaisesRegex(RuntimeError, '3D or 4D'):
            inp = torch.randn(16, 0, 20, 32, device=device)
            avgpool(inp)

    @onlyCUDA
    @largeTensorTest('16GB')
    def test_prelu_backward_32bit_indexing(self, device):
        m = torch.nn.PReLU().cuda().half()
        input_ = torch.ones((1024, 1024, 1024, 2), dtype=torch.half, device=device)
        output = m(input_)
        output.backward(input_)

    def test_linear_empty(self, device):
        mod = torch.nn.Linear(7, 7).to(device)
        inp = torch.randn(0, 7, device=device)
        self._test_module_empty_input(mod, inp)

    def test_one_hot(self, device):
        if self.device_type != 'cuda':  # cuda throws device assert for invalid data
            with self.assertRaises(RuntimeError):
                torch.nn.functional.one_hot(torch.tensor([3, 4, -1, 0], device=device), -1)

            with self.assertRaises(RuntimeError):
                torch.nn.functional.one_hot(torch.tensor([3, 4, 1, 0], device=device), 3)

        t = torch.nn.functional.one_hot(torch.tensor([3, 4, 1, 0], device=device))
        expected = torch.tensor([[0, 0, 0, 1, 0],
                                 [0, 0, 0, 0, 1],
                                 [0, 1, 0, 0, 0],
                                 [1, 0, 0, 0, 0]], device=device)
        self.assertEqual(t, expected)

        t = torch.nn.functional.one_hot(torch.tensor([3, 4, 1, 0], device=device), -1)
        expected = torch.tensor([[0, 0, 0, 1, 0],
                                 [0, 0, 0, 0, 1],
                                 [0, 1, 0, 0, 0],
                                 [1, 0, 0, 0, 0]], device=device)
        self.assertEqual(t, expected)

        t = torch.nn.functional.one_hot(torch.tensor([3, 4, 1, 0], device=device), 6)
        expected = torch.tensor([[0, 0, 0, 1, 0, 0],
                                 [0, 0, 0, 0, 1, 0],
                                 [0, 1, 0, 0, 0, 0],
                                 [1, 0, 0, 0, 0, 0]], device=device)
        self.assertEqual(t, expected)

        t = torch.nn.functional.one_hot(torch.tensor([[3, 4], [1, 0]], device=device))
        expected = torch.tensor([[[0, 0, 0, 1, 0],
                                  [0, 0, 0, 0, 1]],
                                 [[0, 1, 0, 0, 0],
                                  [1, 0, 0, 0, 0]]], device=device)
        self.assertEqual(t, expected)

        t = torch.nn.functional.one_hot(torch.tensor(4, device=device))
        expected = torch.tensor([0, 0, 0, 0, 1], device=device)
        self.assertEqual(t, expected)

        t = torch.nn.functional.one_hot(torch.empty([4, 0], dtype=torch.long, device=device), 100)
        expected = torch.empty([4, 0, 100], dtype=torch.long)
        self.assertEqual(t, expected)

        with self.assertRaises(RuntimeError):
            torch.nn.functional.one_hot(torch.empty([4, 0], dtype=torch.long, device=device))

        with self.assertRaises(RuntimeError):
            torch.nn.functional.one_hot(torch.tensor([3, 4, 1, 0], device=device), -2)

    def test_nn_scalars(self, device):
        # One off tests to ensure scalars from nn.yaml are properly applied
        def verify_scalars(input, output):
            if input.dim() == 0:
                self.assertEqual((), output.shape)
            else:
                self.assertNotEqual((), output.shape)
            output.sum().backward()
            self.assertEqual(input.shape, input.grad.shape)

        for input_shape in [(5, 6), ()]:
            for module in [torch.nn.ELU, torch.nn.Hardtanh, torch.nn.LeakyReLU, torch.nn.LogSigmoid,
                           torch.nn.RReLU, torch.nn.Softshrink, torch.nn.Softplus, torch.nn.Sigmoid,
                           torch.nn.Tanh]:
                input = torch.randn(input_shape, device=device, requires_grad=True)
                m = module()
                output = m(input)
                verify_scalars(input, output)

    def test_nn_scalars_reductions(self, device):
        # One off tests to ensure scalars from nn.yaml are properly applied
        def verify_reduction_scalars(input, reduction, output):
            if reduction != 'none' or input.dim() == 0:
                self.assertEqual((), output.shape)
            else:
                self.assertNotEqual((), output.shape)
            output.sum().backward()
            self.assertEqual(input.shape, input.grad.shape)

        for input_shape in [(5, 6), ()]:
            for reduction in ['none', 'mean', 'sum']:
                for module in [torch.nn.BCELoss, torch.nn.L1Loss, torch.nn.MSELoss,
                               torch.nn.SmoothL1Loss, torch.nn.SoftMarginLoss]:
                    input = torch.randn(input_shape, device=device, requires_grad=True)
                    target = torch.empty(input_shape, device=device).random_(2)
                    sigmoid = nn.Sigmoid()

                    input = torch.randn(input_shape, device=device, requires_grad=True)
                    m = module(reduction=reduction)
                    output = m(sigmoid(input), target)
                    verify_reduction_scalars(input, reduction, output)

    # verify that bogus reduction strings are errors
    @onlyOnCPUAndCUDA
    def test_invalid_reduction_strings(self, device):
        input = torch.randn(3, 5, requires_grad=True, device=device)
        cinput = torch.randn(3, 5, requires_grad=True, device=device, dtype=torch.cfloat)
        target = torch.tensor([1, 0, 4], device=device)
        var = torch.ones(size=input.size(), requires_grad=True, device=device)

        for reduction in ['none', 'invalid']:
            def v(fn):
                if reduction == 'invalid':
                    self.assertRaises(ValueError, lambda: fn())
                else:
                    fn()

            v(lambda: F.nll_loss(input, target, reduction=reduction))
            v(lambda: F.cross_entropy(input, target, reduction=reduction))
            v(lambda: F.multi_margin_loss(input, target, reduction=reduction))

            v(lambda: F.kl_div(input, input, reduction=reduction))
            v(lambda: F.huber_loss(input, input, reduction=reduction))
            v(lambda: F.smooth_l1_loss(input, input, reduction=reduction))
            v(lambda: F.l1_loss(input, input, reduction=reduction))
            v(lambda: F.l1_loss(cinput, cinput, reduction=reduction))
            v(lambda: F.mse_loss(input, input, reduction=reduction))
            v(lambda: F.hinge_embedding_loss(input, input, reduction=reduction))
            v(lambda: F.poisson_nll_loss(input, input, reduction=reduction))
            v(lambda: F.gaussian_nll_loss(input, input, var, reduction=reduction))
            v(lambda: F.binary_cross_entropy(torch.sigmoid(input), input, reduction=reduction))
            v(lambda: F.binary_cross_entropy_with_logits(input, input, reduction=reduction))

            zeros = torch.zeros_like(input).to(torch.int64)
            v(lambda: F.multilabel_soft_margin_loss(input, zeros, reduction=reduction))
            v(lambda: F.multilabel_margin_loss(input, zeros, reduction=reduction))

            v(lambda: F.triplet_margin_loss(input, input, input, reduction=reduction))
            v(lambda: F.triplet_margin_with_distance_loss(input, input, input, reduction=reduction))
            v(lambda: F.margin_ranking_loss(input, input, input.sign(), reduction=reduction))
            v(lambda: F.cosine_embedding_loss(input, input, input[:, 0].sign(), reduction=reduction))

            log_probs = torch.randn(50, 16, 20, requires_grad=True, device=device).log_softmax(2)
            targets = torch.randint(1, 20, (16, 30), dtype=torch.long, device=device)
            input_lengths = torch.full((16,), 50, dtype=torch.long, device=device)
            target_lengths = torch.randint(10, 30, (16,), dtype=torch.long, device=device)
            v(lambda: F.ctc_loss(log_probs, targets, input_lengths, target_lengths, reduction=reduction))

            # FIXME: should we allow derivatives on these?
            v(lambda: F.soft_margin_loss(input, input.sign().detach(), reduction=reduction))

    @onlyOnCPUAndCUDA
    def test_smooth_l1_loss_vs_huber_loss(self, device):
        def _make_test_tensor(shape, contiguous=True):
            if contiguous:
                test_tensor = torch.randn(shape, device=device)
            else:
                # Select every other element in the innermost dimension to
                # make it non-contiguous.
                doubled_shape = list(shape)
                doubled_shape[-1] *= 2
                test_tensor = torch.randn(doubled_shape, device=device)
                test_tensor = test_tensor[..., ::2]
            return test_tensor

        def _test_smooth_l1_loss_vs_huber_loss_helper(input, target, beta, require_equal):
            for reduction in ['mean', 'sum', 'none']:
                smooth_l1 = torch.nn.SmoothL1Loss(beta=beta, reduction=reduction)
                # beta hyper-parameter is called delta for Huber
                huber = torch.nn.HuberLoss(delta=beta, reduction=reduction)
                smooth_l1_loss = smooth_l1(input, target)
                huber_loss = huber(input, target)

                if require_equal:
                    self.assertEqual(smooth_l1_loss, huber_loss)
                else:
                    # Huber loss should be larger than smooth L1 loss by a factor of beta.
                    self.assertEqual(smooth_l1_loss * beta, huber_loss)

        def _test_smooth_l1_loss_vs_huber_loss_multi_input_helper(beta, require_equal):
            # Test the non-vectorized case.
            shape = (2, 2)
            _test_smooth_l1_loss_vs_huber_loss_helper(input=_make_test_tensor(shape),
                                                      target=_make_test_tensor(shape),
                                                      beta=beta,
                                                      require_equal=require_equal)

            # Test the vectorized case (innermost dim > 32).
            shape = (64, 64)
            _test_smooth_l1_loss_vs_huber_loss_helper(input=_make_test_tensor(shape),
                                                      target=_make_test_tensor(shape),
                                                      beta=beta,
                                                      require_equal=require_equal)

            # Test the non-contiguous case.
            _test_smooth_l1_loss_vs_huber_loss_helper(input=_make_test_tensor(shape, contiguous=False),
                                                      target=_make_test_tensor(shape, contiguous=False),
                                                      beta=beta,
                                                      require_equal=require_equal)

        def test_equal_when_beta_is_one():
            _test_smooth_l1_loss_vs_huber_loss_multi_input_helper(beta=1.0, require_equal=True)

        def test_unequal_when_beta_is_less_than_one():
            _test_smooth_l1_loss_vs_huber_loss_multi_input_helper(beta=0.5, require_equal=False)

        def test_unequal_when_beta_is_greater_than_one():
            _test_smooth_l1_loss_vs_huber_loss_multi_input_helper(beta=1.5, require_equal=False)

        test_equal_when_beta_is_one()
        test_unequal_when_beta_is_less_than_one()
        test_unequal_when_beta_is_greater_than_one()

    # We don't want to make propagating NaN a hard requirement on ops, but for
    # these easy ones, we should make them do so.
    def test_nonlinearity_propagate_nan(self, device):
        def test(nonlinearity, *args, **kwargs):
            x = torch.tensor([nan], device=device)
            fn = getattr(F, nonlinearity)
            try:
                self.assertTrue(math.isnan(fn(x, *args, **kwargs).item()))
            except Exception as e:
                if 'not implemented' not in str(e):
                    raise

        test('relu')
        test('relu', inplace=True)
        test('relu6')
        test('elu')
        test('selu')
        test('celu')
        test('rrelu')
        test('rrelu', inplace=True)
        test('hardtanh')
        test('tanh')
        test('sigmoid')
        test('logsigmoid')
        test('hardshrink')
        test('tanhshrink')
        test('softsign')
        test('softmin', 0)
        test('softmax', 0)
        test('log_softmax', 0)
        test('leaky_relu', 0.2)
        test('threshold', 3, 2)
        test('threshold', 3, 2, inplace=True)

    def test_pooling_shape(self, device):
        ''' Test the output shape calculation for pooling functions '''

        # Checks output shape against expected for 1D, 2D and 3D
        def check(expected_out_shape, sizes, *args, **kwargs):
            for kernel in ['max', 'avg']:
                for i in [1, 2, 3]:
                    if hasattr(torch.nn.functional, f'{kernel}_pool{i}d'):
                        op = getattr(torch.nn.functional, f'{kernel}_pool{i}d')
                        t = torch.randn(sizes[:i + 2], device=device)
                        self.assertEqual(op(t, *args, **kwargs).shape, expected_out_shape[:i + 2])

        check((1, 1, 3, 3, 4), (1, 1, 5, 6, 7), kernel_size=1, stride=2, padding=0, ceil_mode=True)
        check((1, 1, 2, 3, 3), (1, 1, 3, 4, 5), kernel_size=2, stride=2, padding=1, ceil_mode=False)
        check((1, 1, 2, 3, 3), (1, 1, 3, 4, 5), kernel_size=2, stride=2, padding=1, ceil_mode=True)

        # Test case from issue https://github.com/pytorch/pytorch/issues/45357
        x = torch.randn(1, 1, 6, 7, device=device)
        y = torch.nn.functional.max_pool2d(x, 1, stride=(2, 2), padding=0, ceil_mode=True)
        self.assertEqual(y.size(), (1, 1, 3, 4))

    @onlyOnCPUAndCUDA   # TODO: fix on XLA
    def test_adaptive_avg_pool2d_output_size_one(self, device):
        def helper(size, memory_format):
            x = torch.randint(1, 10, size, dtype=torch.float, device=device, requires_grad=True)
            if memory_format == 'non_contiguous':
                x = x[::2, ::2, ::2, ::2]
            else:
                x = x.to(memory_format=memory_format)

            net = torch.nn.AdaptiveAvgPool2d((1, 1))
            out = net(x)
            ref_out = x.contiguous().mean((-1, -2)).view((x.size(0), x.size(1), 1, 1))

            out.sum().backward()    # make sure it doesn't crash

            self.assertEqual(out, ref_out)
            if memory_format == torch.channels_last:
                self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
                c = out.size(1)
                self.assertEqual(out.stride(), [c, 1, c, c])
            else:
                self.assertTrue(out.is_contiguous())
                c = out.size(1)
                self.assertEqual(out.stride(), [c, 1, 1, 1])

        for mf in (torch.contiguous_format, torch.channels_last, 'non_contiguous'):
            helper((2, 3, 6, 6), mf)

    @onlyOnCPUAndCUDA
    def test_adaptive_avg_pool3d_output_size_one(self, device):
        x = torch.randn((2, 3, 6, 6, 6) , dtype=torch.float, device=device, requires_grad=True)

        net = torch.nn.AdaptiveAvgPool3d(1)
        out = net(x)
        ref_out = x.contiguous().mean((-1, -2, -3)).view(out.shape)

        out.sum().backward()    # make sure it doesn't crash

        self.assertEqual(out, ref_out)
        self.assertTrue(out.is_contiguous())
        c = out.size(1)
        self.assertEqual(out.stride(), [c, 1, 1, 1, 1])


    @onlyOnCPUAndCUDA
    @dtypes(torch.uint8, torch.int8, torch.short, torch.int, torch.long)
    def test_adaptive_pooling_no_suppot_input(self, device, dtype):
        for numel in (2, 3):
            for pool_type in ('Max', 'Avg'):
                cls_name = 'Adaptive{}Pool{}d'.format(pool_type, numel)
                module_cls = getattr(nn, cls_name)
                output_size = (2,) * numel
                module = module_cls(output_size)
                input = torch.randn((4,) * (numel + 1), device=device).to(dtype)
                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    output = module(input)

    @onlyOnCPUAndCUDA
    @dtypes(torch.float, torch.double)
    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    def test_avg_pool2d_nhwc(self, device, dtype):
        def helper(n, c, h, w, kernel_size, stride=None,
                   count_include_pad=True, divisor_override=None, padding=0):
            if stride is None:
                stride = kernel_size
            input = torch.randn(n, c, h, w, dtype=dtype, device=device)
            input = input.contiguous(memory_format=torch.channels_last).requires_grad_()
            grad = torch.randn(n, c, (h - kernel_size) // stride + 1, (w - kernel_size) // stride + 1,
                               dtype=dtype, device=device)
            pool = torch.nn.AvgPool2d(kernel_size, stride=stride, count_include_pad=count_include_pad,
                                      divisor_override=divisor_override).to(device)

            ref_input = input.detach().clone().contiguous().requires_grad_(True)
            ref_grad = grad.detach().clone().contiguous()
            ref_pool = torch.nn.AvgPool2d(kernel_size, stride=stride, count_include_pad=count_include_pad,
                                          divisor_override=divisor_override).to(device)

            out = pool(input)
            out.backward(grad)
            ref_out = ref_pool(ref_input)
            ref_out.backward(ref_grad)

            self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
            self.assertTrue(ref_out.is_contiguous())
            self.assertTrue(torch.allclose(out, ref_out))
            self.assertTrue(torch.allclose(input.grad, ref_input.grad))

        helper(4, 8, 8, 8, 3)
        helper(4, 8, 8, 8, 3, count_include_pad=False, padding=1)
        helper(4, 8, 8, 8, 3, count_include_pad=False, padding=2, stride=2)
        helper(4, 8, 8, 8, 3, divisor_override=42)
        helper(4, 8, 8, 8, 7)
        helper(200, 512, 28, 28, 2)
        helper(4, 8, 7, 7, 3, stride=1)
        helper(4, 8, 7, 7, 3, padding=2, stride=1)
        helper(10, 512, 31, 31, 3, stride=2)
        helper(1, 129, 8, 8, 3, stride=2)

    @onlyCPU
    @dtypes(torch.float)
    def test_max_pool1d_errors(self, device, dtype):
        def check(x, args, message):
            model = torch.nn.MaxPool1d(*args)
            with self.assertRaisesRegex(RuntimeError, r'max_pool1d\(\) ' + message):
                model(torch.tensor(x, device=device, dtype=dtype))

        # Pooling args: (kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        check(0, (1,), "input tensor must have 2 or 3 dimensions but got 0")
        check([], (1,), "input tensor must have 2 or 3 dimensions but got 1")
        check([[]], (1, 0), "stride must be greater than zero, but got 0")
        check([[]], (1, 1, -1), "padding must be non-negative, but got -1")
        check([[]], (1, 1, 2), "padding should be at most half of kernel size, but got padding=2 and kernel_size=1")
        check([[]], (1, 1, 0, 0), "dilation must be greater than zero, but got 0")
        check([[]], (5, 1, 0, 1), "Invalid computed output size: -4")

    @onlyCPU
    @dtypes(torch.float, torch.double)
    def test_max_pool1d_corner_cases(self, device, dtype):
        def check(x, args, expected):
            model = torch.nn.MaxPool1d(*args)
            if isinstance(x, list):
                x = torch.tensor(x, device=device, dtype=dtype)
                expected = torch.tensor(expected, device=device, dtype=dtype)
            self.assertEqual(model(x), expected)

        # Pooling args: (kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        check([[]], (1, None, 0, 1, False, False), [[]])
        check([[[]]], (1, None, 0, 1, False, False), [[[]]])
        check([[[]]], (2, 1, 1, 2, False, True), [[[]]])
        check([[1]], (1, None, 0, 1, False, False), [[1]])
        check([[1]], (2, None, 1, 2, False, False), [[float('-inf')]])
        check([[1], [1]], (2, None, 1, 2, False, False), [[float('-inf')], [float('-inf')]])
        check([[1, 2]], (2, 1, 1, 2, False, False), [[2, 1]])
        check([[1, 2]], (2, 2, 1, 2, False, True), [[2, 2]])

        empty_tensor = torch.empty((2, 0, 1), device=device, dtype=dtype)
        check(empty_tensor, (1, None, 0, 1, False, False), empty_tensor)

    @onlyCPU
    @dtypes(torch.float, torch.double)
    def test_max_pool1d(self, device, dtype):
        # FIXME For now compare against max_pool1d with indices
        def check(x, *args, **kwargs):
            model = torch.nn.MaxPool1d(*args, **kwargs)
            ref_model = torch.nn.MaxPool1d(*args, **kwargs, return_indices=True)
            self.assertEqual(model(x), ref_model(x)[0])

        sizes = [random.sample(range(8, 128), 3) for _ in range(3)]
        kernel_sizes = random.sample(range(1, 5), 3)
        strides = random.sample(range(1, 5), 3)
        dilations = random.sample(range(1, 5), 3)
        ceil_modes = [True, False]

        for size, kernel_size, stride, dilation, ceil_mode in \
                itertools.product(sizes, kernel_sizes, strides, dilations, ceil_modes):
            padding = random.sample(range(0, math.floor(kernel_size / 2) + 1), 1)
            check(torch.randn(size, device=device, dtype=dtype),
                  kernel_size, stride, padding, dilation, ceil_mode=ceil_mode)

        # Non-contiguous test
        tensor = torch.randn(5, 151, 33, device=device, dtype=dtype)[::2, ::3, ::2]
        check(tensor, 3, 2, 1, 2, ceil_mode=True)
        check(tensor.transpose(1, 2), 3, 2, 1, 2, ceil_mode=True)

    @onlyCUDA
    def test_max_pool2d(self, device):
        def helper(n, c, h, w, ks):
            x = torch.randn(n, c, h, w, device='cuda', dtype=torch.float, requires_grad=True)
            ref_x = x.detach().clone().cpu().requires_grad_()

            pool = torch.nn.MaxPool2d(kernel_size=ks)

            y = pool(x)
            ref_y = pool(ref_x)

            y.sum().backward()
            ref_y.sum().backward()

            self.assertEqual(y, ref_y)
            self.assertEqual(x.grad, ref_x.grad)

        helper(2, 8, 4, 4, ks=2)
        helper(1, 100000, 32, 32, ks=4)
        helper(1, 100000, 1, 4, ks=(1, 4))  # test for max_pool1d

    @onlyOnCPUAndCUDA
    @dtypes(torch.float, torch.double)
    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    def test_max_pool2d_nhwc(self, device, dtype):
        def helper(n, c, h, w, kernel_size, stride=None):
            if stride is None:
                stride = kernel_size
            input = torch.randn(n, c, h, w, dtype=dtype, device=device)
            input = input.contiguous(memory_format=torch.channels_last).requires_grad_()
            grad = torch.randn(n, c, (h - kernel_size) // stride + 1, (w - kernel_size) // stride + 1,
                               dtype=dtype, device=device)
            pool = torch.nn.MaxPool2d(kernel_size, stride, return_indices=True).to(device)

            ref_input = input.detach().clone().contiguous().requires_grad_(True)
            ref_grad = grad.detach().clone().contiguous()
            ref_pool = torch.nn.MaxPool2d(kernel_size, stride, return_indices=True).to(device)

            out, ind = pool(input)
            out.backward(grad)
            ref_out, ref_ind = ref_pool(ref_input)
            ref_out.backward(ref_grad)

            self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
            self.assertTrue(ref_out.is_contiguous())
            self.assertTrue(ind.is_contiguous(memory_format=torch.channels_last))
            self.assertTrue(ref_ind.is_contiguous())
            self.assertTrue(torch.allclose(out, ref_out))
            self.assertTrue(torch.allclose(ind, ref_ind))
            self.assertTrue(torch.allclose(input.grad, ref_input.grad))

        helper(4, 8, 8, 8, 7)
        helper(200, 512, 28, 28, 2)
        helper(4, 8, 7, 7, 3, stride=1)
        helper(10, 512, 31, 31, 3, stride=2)
        helper(1, 129, 8, 8, 3, stride=2)

    @onlyCUDA
    def test_max_pool2d_indices(self, device):
        def helper(n, c, h, w, ks):
            if n is None:
                x = torch.randn(c, h, w, device='cuda', dtype=torch.float, requires_grad=True)
            else:
                x = torch.randn(n, c, h, w, device='cuda', dtype=torch.float, requires_grad=True)

            ref_x = x.detach().clone().cpu().requires_grad_()

            pool = torch.nn.MaxPool2d(kernel_size=ks, return_indices=True)

            y, idx = pool(x)
            ref_y, ref_idx = pool(ref_x)

            y.sum().backward()
            ref_y.sum().backward()

            self.assertEqual(y, ref_y)
            self.assertEqual(idx, ref_idx)  # assertEqual implicitly compares shape for tensors
            self.assertEqual(x.grad, ref_x.grad)

        helper(2, 8, 4, 4, ks=2)
        helper(None, 3, 50, 50, ks=5)

    def test_upsamplingNearest2d(self, device):
        for memory_format in [torch.contiguous_format, torch.channels_last]:
            in_t = torch.ones(1, 2, 2, 2, device=device).contiguous(memory_format=memory_format)
            in_uint8_t = torch.ones(1, 2, 2, 2, dtype=torch.uint8, device=device).contiguous(memory_format=memory_format)
            with warnings.catch_warnings(record=True) as w:
                out_t = F.interpolate(in_t, size=4, mode='nearest')
                out_uint8_t = F.interpolate(in_uint8_t, size=4, mode='nearest')
            self.assertEqual(torch.ones(1, 2, 4, 4, device=device), out_t)
            self.assertEqual(torch.ones(1, 2, 4, 4, dtype=torch.uint8, device=device), out_uint8_t)
            # Assert that memory format is carried through to the output
            self.assertTrue(out_t.is_contiguous(memory_format=memory_format))

            # test forward when input's height is not same as width
            in_t = torch.ones(1, 2, 2, 1, device=device).contiguous(memory_format=memory_format).requires_grad_()
            with warnings.catch_warnings(record=True) as w:
                out_t = F.interpolate(in_t, size=(4, 2), mode='nearest')
            self.assertEqual(torch.ones(1, 2, 4, 2, device=device), out_t)
            self.assertTrue(out_t.is_contiguous(memory_format=memory_format))

            out_t.backward(torch.randn_like(out_t))
            self.assertTrue(in_t.grad.is_contiguous(memory_format=memory_format))

            # test backward when input's height is not same as width
            input = torch.ones(1, 2, 2, 1, requires_grad=True, device=device).contiguous(memory_format=memory_format)
            gradcheck(lambda x: F.interpolate(x, size=(4, 2), mode='nearest'), [input])
            gradgradcheck(lambda x: F.interpolate(x, size=(4, 2), mode='nearest'), [input])

            input = torch.randn(1, 2, 2, 2, requires_grad=True, device=device).contiguous(memory_format=memory_format)
            self.assertEqual(
                F.interpolate(input, 4, mode='nearest'),
                F.interpolate(input, scale_factor=2, mode='nearest'))
            gradcheck(lambda x: F.interpolate(x, 4, mode='nearest'), [input])
            gradgradcheck(lambda x: F.interpolate(x, 4, mode='nearest'), [input])

            # Assert that cpu and cuda handle channels_last memory format in the same way
            # https://github.com/pytorch/pytorch/issues/54590
            if torch.device(device).type == 'cuda':
                for shapes, scale_factor in product([
                    (2, 2, 3, 4), (2, 3, 4, 5), (3, 1, 2, 2), (1, 5, 3, 2)
                ], [0.5, 1.5, 2]):
                    a_cuda = torch.randn(*shapes, device=device).contiguous(memory_format=memory_format).requires_grad_()
                    a_cpu = a_cuda.detach().cpu().requires_grad_()

                    with warnings.catch_warnings(record=True):
                        out_cuda = F.interpolate(a_cuda, scale_factor=scale_factor, mode='nearest')
                        out_cpu = F.interpolate(a_cpu, scale_factor=scale_factor, mode='nearest')

                    self.assertEqual(out_cpu.cuda(), out_cuda)

                    g_cuda = torch.randn_like(out_cuda)
                    g_cpu = g_cuda.cpu()

                    out_cuda.backward(g_cuda)
                    out_cpu.backward(g_cpu)

                    self.assertEqual(a_cuda.grad, a_cpu.grad)

    def test_upsamplingBilinear2d(self, device):
        for align_corners in [True, False]:
            kwargs = dict(mode='bilinear', align_corners=align_corners)
            for memory_format in [torch.contiguous_format, torch.channels_last]:
                # test float scale factor up & downsampling
                for scale_factor in [0.5, 1.5, 2]:
                    in_t = torch.ones(1, 2, 2, 2, device=device).contiguous(memory_format=memory_format).requires_grad_()
                    out_size = int(math.floor(in_t.shape[-1] * scale_factor))
                    with warnings.catch_warnings(record=True) as w:
                        out_t = F.interpolate(in_t, scale_factor=scale_factor, **kwargs)
                    self.assertEqual(torch.ones(1, 2, out_size, out_size, device=device), out_t.data)
                    # Assert that memory format is carried through to the output
                    self.assertTrue(out_t.is_contiguous(memory_format=memory_format))
                    out_t.backward(torch.randn_like(out_t))
                    self.assertTrue(in_t.grad.is_contiguous(memory_format=memory_format))

                    input = torch.randn(1, 2, 2, 2, device=device).contiguous(memory_format=memory_format).requires_grad_()
                    gradcheck(lambda x: F.interpolate(x, out_size, **kwargs), [input])

                    # Assert that cpu and cuda give same results
                    if torch.device(device).type == 'cuda':
                        for shapes in [
                            (2, 2, 3, 4), (2, 3, 4, 5), (3, 1, 2, 2), (1, 5, 3, 2)
                        ]:
                            a_cuda = torch.randn(*shapes, device=device).contiguous(memory_format=memory_format).requires_grad_()
                            a_cpu = a_cuda.detach().cpu().requires_grad_()

                            with warnings.catch_warnings(record=True):
                                out_cuda = F.interpolate(a_cuda, scale_factor=scale_factor, **kwargs)
                                out_cpu = F.interpolate(a_cpu, scale_factor=scale_factor, **kwargs)

                            self.assertEqual(out_cpu.cuda(), out_cuda)

                            g_cuda = torch.randn_like(out_cuda)
                            g_cpu = g_cuda.cpu()

                            out_cuda.backward(g_cuda)
                            out_cpu.backward(g_cpu)

                            self.assertEqual(a_cuda.grad, a_cpu.grad)

    @onlyCPU
    @dtypes(torch.float, torch.double)
    def test_adaptive_pooling_max_nhwc(self, device, dtype):
        def helper(n, c, h, w, output_height, output_width, contig):
            input = torch.randint(1, 10, (n, c, h, w), device=device, dtype=dtype)
            input = input.contiguous(memory_format=torch.channels_last)
            grad = torch.randint(1, 10, (4, 8, output_height, output_width), device=device, dtype=dtype)
            grad = grad.contiguous(memory_format=torch.channels_last)
            if not contig:
                input = input[:, ::2, :, :]
                grad = grad[:, ::2, :, :]
            input.requires_grad_(True)
            pool = torch.nn.AdaptiveMaxPool2d((output_height, output_width), return_indices=True).to(device)

            ref_input = input.detach().clone().contiguous().requires_grad_(True)
            ref_grad = grad.detach().clone().contiguous()
            ref_pool = torch.nn.AdaptiveMaxPool2d((output_height, output_width), return_indices=True).to(device)

            out, ind = pool(input)
            out.backward(grad)
            ref_out, ref_ind = ref_pool(ref_input)
            ref_out.backward(ref_grad)

            self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
            self.assertTrue(ref_out.is_contiguous())
            self.assertTrue(ind.is_contiguous(memory_format=torch.channels_last))
            self.assertTrue(ref_ind.is_contiguous())
            self.assertEqual(out, ref_out)
            self.assertEqual(ind, ref_ind)
            self.assertEqual(input.grad, ref_input.grad)

        for contig in [True, False]:
            helper(4, 8, 10, 10, 7, 7, contig)
            helper(4, 8, 9, 14, 5, 8, contig)
            helper(4, 8, 11, 11, 1, 1, contig)

    def test_embedding_dense_grad(self, device):
        embd = nn.Embedding(20, 20).to(device)
        weight = embd.weight

        def fn_wrapper(device):
            def fn(weight):
                inp = torch.tensor([[0, 1, 1, 2], [3, 5, 7, 11]], dtype=torch.long).to(device)
                return torch.nn.functional.embedding(inp, weight)
            return fn

        fn = fn_wrapper(device)
        _assertGradAndGradgradChecks(self, fn, (weight, ))

    def test_embedding_scalar_weight_error(self, device):
        indices = torch.rand(2, 2, device=device).long()
        weights = [
            torch.tensor(1.0, device=device),
            torch.tensor(1.0, device=device).reshape(1, 1, 1),
        ]

        for weight in weights:
            with self.assertRaisesRegex(RuntimeError, "'weight' must be 2-D"):
                torch.nn.functional.embedding(indices, weight)

    @dtypesIfCUDA(torch.float16, torch.float64)
    @dtypes(torch.float64)
    def test_embedding_backward(self, device, dtype):
        embedding = nn.Embedding(10, 3, sparse=True)
        tensor = torch.tensor([[7, 1, 3]])
        ones = torch.tensor(1., dtype=dtype).expand(3, 3)
        tensorTwice = tensor.repeat(1, 2)
        onesTwice = torch.cat((ones, ones))

        embedding = embedding.to(dtype=dtype).to(device)
        tensor = tensor.to(device)
        ones = ones.to(device)
        tensorTwice = tensorTwice.to(device)
        onesTwice = onesTwice.to(device)

        embedding.zero_grad()
        embedding(tensor[0]).sum().backward()
        self.assertEqual(embedding.weight.grad._indices(), tensor)
        self.assertEqual(embedding.weight.grad._values(), ones)

        embedding.zero_grad()
        embedding(tensor[0]).sum().backward()
        embedding(tensor[0]).sum().backward()
        self.assertEqual(embedding.weight.grad._indices(), tensorTwice)
        self.assertEqual(embedding.weight.grad._values(), onesTwice)

        embedding.zero_grad()
        embedding(tensor[0]).sum().backward()
        tensor[0, 0] = 8
        embedding(tensor[0]).sum().backward()
        tensorTwice[0, 3] = 8
        self.assertEqual(embedding.weight.grad._indices(), tensorTwice)
        self.assertEqual(embedding.weight.grad._values(), onesTwice)

    @dtypesIfCUDA(*ALL_TENSORTYPES2)
    @dtypes(torch.float32)
    def test_embedding_padding_idx(self, device, dtype):
        embedding = nn.Embedding(10, 20, padding_idx=0).to(device, dtype)
        input = torch.tensor([[0, 2, 4, 5], [4, 3, 0, 9]], dtype=torch.long).to(device)
        output = embedding(input)
        self.assertEqual(output[0][0].sum(), 0)
        self.assertEqual(output[1][2].sum(), 0)

        embedding = nn.Embedding(10, 20, padding_idx=0, sparse=True).to(device, dtype)
        input = torch.tensor([[0, 2, 4, 5], [4, 3, 0, 9]], dtype=torch.long).to(device)
        output = embedding(input)
        self.assertEqual(output[0][0].sum(), 0)
        self.assertEqual(output[1][2].sum(), 0)

        # negative indexing check for padding_idx
        # padding_idx=-2, num_embeddings=10 ==> index 8 padded
        embedding = nn.Embedding(10, 20, padding_idx=-2).to(device, dtype)
        input = torch.tensor([[0, 2, 8, 5], [4, 8, 0, 9]], dtype=torch.long).to(device)
        output = embedding(input)
        self.assertEqual(output[0][2].sum(), 0)
        self.assertEqual(output[1][1].sum(), 0)

        embedding = nn.Embedding(10, 20, padding_idx=-2, sparse=True).to(device, dtype)
        input = torch.tensor([[0, 2, 8, 5], [4, 8, 0, 9]], dtype=torch.long).to(device)
        output = embedding(input)
        self.assertEqual(output[0][2].sum(), 0)
        self.assertEqual(output[1][1].sum(), 0)

        # change padding vector
        padding_vector = torch.ones(20, dtype=dtype, device=device)
        embedding = nn.Embedding(10, 20, padding_idx=2, sparse=True).to(device, dtype)
        with torch.no_grad():
            embedding.weight[2] = padding_vector
        input = torch.tensor([0, 2], dtype=torch.long).to(device)
        output = embedding(input)
        self.assertEqual(output[1], padding_vector)

        # out of bounds check for padding_idx
        self.assertRaises(AssertionError, nn.Embedding, num_embeddings=10, embedding_dim=20, padding_idx=25)
        self.assertRaises(AssertionError, nn.Embedding, num_embeddings=10, embedding_dim=20, padding_idx=-25)

        padding_idx = 0
        embedding = nn.Embedding(5, 2, padding_idx=padding_idx).to(device, dtype)
        for n in (1, 2, 1000):  # Need large N to trigger all the methods we have implemented
            for other_indices in ([], [1, 3], [2]):
                indices = torch.tensor(other_indices + [padding_idx] * n, dtype=torch.long).to(device)
                pre = embedding.weight[padding_idx].clone()
                embedding(indices).sum().backward()
                after = (embedding.weight + embedding.weight.grad)[padding_idx]
                embedding.zero_grad()
                self.assertEqual(after, pre)

                # test double backward
                emb_sum = embedding(indices).sum()
                emb_grad = torch.autograd.grad(outputs=emb_sum, inputs=list(embedding.parameters()), retain_graph=True)
                scalar = emb_grad[0].sum() + emb_sum
                scalar.backward()
                after = (embedding.weight + embedding.weight.grad)[padding_idx]
                embedding.zero_grad()
                self.assertEqual(after, pre)

    # Check correctness of torch.nn.functional.embedding_bag forward and
    # backward functions with padding_idx, given a 1D input separated into bags
    # with an offset array. Compare against an equivalent 2D input that uses
    # padding indices to fill in the gaps indicated by the offset array

    @onlyOnCPUAndCUDA
    @dtypes(torch.float32, torch.float64)
    @dtypesIfCUDA(torch.half, torch.bfloat16)
    def test_embedding_bag_1D_padding_idx(self, device, dtype):
        num_features = 3
        max_indices_per_bag = 10
        num_bags = 10
        num_words = 100

        def gen_1D_indices_offsets(include_last_offset, allpad):
            indices = []
            offsets = []
            cur_offset = 0

            # Make one bag full and one bag empty, for extra coverage
            empty_bag = random.randint(0, num_bags - 1)
            full_bag = empty_bag
            while full_bag == empty_bag:
                full_bag = random.randint(0, num_bags - 1)

            for bag in range(num_bags):
                offsets.append(cur_offset)
                if bag == full_bag:
                    bag_size = max_indices_per_bag
                elif bag == empty_bag:
                    bag_size = 0
                else:
                    bag_size = random.randint(1, max_indices_per_bag - 1)
                indices += [1 if allpad else random.randint(0, num_words - 1) for _ in range(bag_size)]
                cur_offset += bag_size

            # embedding_bag requires first entry of offsets to be 0
            assert offsets[0] == 0

            indices = torch.tensor(indices, device=device)

            if include_last_offset:
                offsets.append(indices.size(0))

            offsets = torch.tensor(offsets, device=device)

            return indices, offsets

        # Convert a 1-D indices-offsets representation into 2-D. Fill any empty
        # indices with padding_idx
        def gen_2D_indices_from_1D(indices_1D, offsets, include_last_offset, padding_idx):
            assert offsets[0] == 0
            if include_last_offset:
                offsets = offsets[:-1]
            indices_2D = torch.empty(num_bags, max_indices_per_bag, device=device, dtype=torch.long)
            for bag in range(num_bags):
                # Determine the start and end position of the bag within indices_1D
                start = offsets[bag]
                end = len(indices_1D) if bag + 1 == num_bags else offsets[bag + 1]
                end = min(len(indices_1D), end)

                # Pull out the bag's indices from indices_1D, and fill any
                # remaining space with padding indices
                indices_in_bag = []
                for item_pos in range(0, max_indices_per_bag):
                    if (start + item_pos) < end:
                        indices_in_bag.append(indices_1D[start + item_pos])
                    else:
                        indices_in_bag.append(padding_idx)
                indices_2D[bag] = torch.tensor(indices_in_bag, device=device)

            return indices_2D

        test_cases = product(['max', 'mean', 'sum'], [False, True], [False, True], [False, True])

        for mode, sparse, include_last_offset, allpad in test_cases:
            # Max sparse and bfloat16 are not supported
            if mode == 'max':
                if sparse or (dtype == torch.bfloat16):
                    continue
            indices_1D, offsets = gen_1D_indices_offsets(include_last_offset, allpad)
            for padding_idx_1D in list(set(indices_1D.tolist())) + [None]:
                msg = (
                    f"mode: '{mode}', sparse: {sparse}, include_last_offset: {include_last_offset}, "
                    f"padding_idx_1D: {padding_idx_1D}")

                # If 1D input does not use a padding index, we still need one for the 2D input,
                # so we can add one dummy word to the weights to act as the padded word
                padding_idx_2D = padding_idx_1D if padding_idx_1D is not None else num_words
                num_words_with_padding = num_words if padding_idx_1D is not None else num_words + 1

                indices_2D = gen_2D_indices_from_1D(
                    indices_1D,
                    offsets,
                    include_last_offset,
                    padding_idx_2D)

                weights = torch.randn(
                    num_words_with_padding,
                    num_features,
                    dtype=dtype,
                    device=device,
                    requires_grad=True)
                weights_check = weights.clone().detach().requires_grad_(True)

                bag = torch.nn.functional.embedding_bag(
                    indices_1D,
                    weights,
                    offsets,
                    padding_idx=padding_idx_1D,
                    mode=mode,
                    sparse=sparse,
                    include_last_offset=include_last_offset)

                bag_check = torch.nn.functional.embedding_bag(
                    indices_2D,
                    weights_check,
                    padding_idx=padding_idx_2D,
                    mode=mode,
                    sparse=sparse)
                self.assertEqual(bag, bag_check, msg=msg)

                bag.sum().backward()
                bag_check.sum().backward()

                # Sometimes, half dtype gradients mismatch by a greater amount
                # than other dtypes
                if dtype in [torch.half, torch.bfloat16]:
                    atol = 0.01
                    rtol = 0.01
                else:
                    atol = None
                    rtol = None
                self.assertEqual(weights.grad, weights_check.grad, msg=msg, atol=atol, rtol=rtol)

    # Check correctness of torch.nn.functional.embedding_bag forward and
    # backward functions with padding_idx, given a 2D indices input. Compare
    # against torch.nn.functional.embedding followed by a reduction.
    @onlyOnCPUAndCUDA
    @dtypes(torch.float32, torch.float64)
    @dtypesIfCUDA(torch.half, torch.bfloat16)
    def test_embedding_bag_2D_padding_idx(self, device, dtype):
        # Use a Python implementation of embedding_bag with padding_idx support
        # to check torch.nn.functional.embedding_bag correctness
        def embedding_bag_check(indices, weights, mode, sparse, padding_idx):
            assert padding_idx is not None
            embedding = torch.nn.functional.embedding(
                indices,
                weights,
                padding_idx=padding_idx,
                sparse=sparse)

            reduction_dim = indices.dim() - 1

            if mode == 'sum' or mode == 'mean':
                # We must avoid including elements at padding_idx in the
                # sum/mean, so multiply those elements by 0, and multiply
                # all other elements by 1
                per_sample_weights = indices.ne(padding_idx).to(dtype).unsqueeze(-1)
                res = embedding.mul(per_sample_weights).sum(dim=reduction_dim)

                if mode == 'mean':
                    weights_sum = per_sample_weights.sum(dim=reduction_dim)
                    res = res.div(weights_sum)

            elif mode == 'max':
                # We must avoid allowing elements at padding_idx to be chosen
                # as the max, so set those elements to negative infinity
                res = embedding.masked_fill(
                    indices.unsqueeze(-1) == padding_idx, -float('inf')
                ).amax(dim=reduction_dim)

            else:
                raise RuntimeError(f"mode '{mode}' is not available")

            # If a row is all padding, set its corresponding result row to 0.
            # This is needed because the above mean and max mode
            # implementations set these elements to nan and -inf, respectively
            if mode in ['mean', 'max']:
                res = res.masked_fill(
                    indices.eq(padding_idx).all(dim=-1).unsqueeze(-1),
                    0)

            return res

        num_features = 3
        num_words = 10
        indices_dim1 = 10

        for mode, sparse, allpad, indices_dim0 in product(['max', 'mean', 'sum'], [False, True], [False, True], [1, 10]):
            # Max sparse and bfloat16 are not supported
            if mode == 'max':
                if sparse or (dtype == torch.bfloat16):
                    continue

            if allpad:
                indices = torch.empty(indices_dim0, indices_dim1, dtype=torch.long, device=device).fill_(1)
            else:
                indices = torch.randint(0, num_words, (indices_dim0, indices_dim1), device=device)

                if indices_dim0 > 1:
                    # Fill one row with duplicate index so we can test with a fully
                    # padded row
                    duplicate_row = random.randint(0, indices_dim0 - 1)
                    indices[duplicate_row] = indices[duplicate_row][0]

            for padding_idx in list(set(indices.flatten(0, -1).tolist())):
                weights = torch.randn(num_words, num_features, dtype=dtype, device=device, requires_grad=True)
                weights_check = weights.clone().detach().requires_grad_(True)

                msg = (
                    f"mode: '{mode}', sparse: {sparse}, padding_idx: {padding_idx}, "
                    f"allpad: {allpad}, indices.size(): {indices.size()}")

                # Check forward with a Python implementation of padding_idx embedding_bag
                bag_check = embedding_bag_check(
                    indices,
                    weights_check,
                    mode,
                    sparse,
                    padding_idx)
                bag = torch.nn.functional.embedding_bag(
                    indices,
                    weights,
                    padding_idx=padding_idx,
                    mode=mode,
                    sparse=sparse)

                self.assertEqual(bag, bag_check, msg=msg)

                bag_check.sum().backward()
                grad_check = weights_check.grad

                bag.sum().backward()
                grad = weights.grad

                # Sometimes, half dtype gradients mismatch by a greater amount
                # than other dtypes
                if dtype in [torch.half, torch.bfloat16]:
                    atol = 0.01
                    rtol = 0.01
                else:
                    atol = None
                    rtol = None
                self.assertEqual(grad, grad_check, msg=msg, atol=atol, rtol=rtol)

    # Test fails on Vg20
    @skipCUDAIfRocm
    @dtypesIfCUDA(torch.half, torch.float)
    @dtypes(torch.float)
    def test_softmax_results(self, device, dtype):
        # Non-even sizes and non-zero shifts test fallback paths in vectorized kernel
        # Note: dim1 > 1024 is needed to exercise the vectorized (non-persistent) path, (16, 30576) is BERT-esque
        sizes = [(0, 10), (32, 20), (10, 0), (31, 20), (32, 21), (31, 23), (32, 1536), (31, 2048), (33, 2049), (16, 30576)]
        shifts = [(0, 0), (1, 0), (0, 1), (1, 1)]
        for fn in [F.softmax, F.log_softmax]:
            for size in sizes:
                for shift in shifts:
                    input = torch.rand(size, device=device, dtype=dtype)
                    # Note: With the largest tests we can hit upper limit of fp16 when we
                    # sum, so scale the input down to stay in a nicer range.
                    if dtype == torch.float16:
                        input = input / 100.
                    input = input[shift[0]:, shift[1]:]
                    # Note; Don't want to bprop back through slice op
                    input = input.detach().requires_grad_(True)
                    ref_input = input.clone().cpu().detach().requires_grad_(True)
                    for dim in [0, 1]:
                        ref_output = fn(ref_input, dtype=torch.float, dim=dim)
                        output = fn(input, dtype=torch.float, dim=dim)
                        grad_output = torch.rand(size, device=device, dtype=dtype)
                        grad_output = grad_output[shift[0]:, shift[1]:]
                        ref_grad_output = grad_output.clone().cpu().detach()
                        grad_input, = torch.autograd.grad(output, input, grad_outputs=(grad_output), create_graph=True)
                        ref_grad_input, = torch.autograd.grad(ref_output, ref_input,
                                                              grad_outputs=(ref_grad_output), create_graph=True)
                        grad_input.sum().backward()
                        ref_grad_input.sum().backward()

                        self.assertEqual(output, ref_output)
                        self.assertEqual(grad_input, ref_grad_input)
                        self.assertEqual(input.grad, ref_input.grad)

    @onlyCUDA
    @dtypesIfCUDA(torch.float, torch.half)
    @largeTensorTest("20GB")
    @precisionOverride({torch.half: 0.001})
    def test_softmax_64bit_indexing(self, device, dtype):
        def run_test(*shape):
            x = torch.randn(shape, device="cuda", dtype=torch.float16, requires_grad=True)
            y = F.log_softmax(x, dim=-1, dtype=dtype)
            y.backward(y)
            with torch.no_grad():
                xx = x.cpu().requires_grad_()
            yy = F.log_softmax(xx.float(), dim=-1).to(dtype)
            yy.backward(yy)
            self.assertEqual(y, yy)
            self.assertEqual(x.grad, xx.grad)

        run_test(1100000000, 2)  # Illegal memory access https://github.com/pytorch/pytorch/issues/52715
        run_test(2200000000, 1)  # invalid configuration argument https://github.com/pytorch/pytorch/issues/52716

    @dtypes(torch.float)
    @dtypesIfCUDA(torch.float, torch.half)
    def test_log_softmax_big(self, device, dtype):
        def _test_helper(shape):
            # generate a tensor with big numbers that are exactly representable in dtype
            # and are at a constant offset from tensor with small numbers
            # the logsoftmax of a small and big tensors should be equal
            x_small = torch.randint(100, shape, dtype=dtype, device=device)
            offset = 1.5e3 if dtype == torch.half else 1e7
            x_big = x_small + offset
            self.assertEqual(F.log_softmax(x_small, -1), F.log_softmax(x_big, -1))
        _test_helper((16, 4))
        if self.device_type == 'cuda':
            # test non-persistent softmax kernel
            _test_helper((4, 1536))

    @onlyCUDA
    @largeTensorTest('12GB')
    def test_conv_large_nosplit(self, device):
        # Here we just test the convolution correctly route to the fallback implementation
        # that is, it does not crash. The correctness of fallback implementation should be
        # covered in other tests
        dtype = torch.half if self.device_type == 'cuda' else torch.float
        conv1 = nn.Conv2d(2, 2, 8, 8).to(device).to(dtype)
        input_large = torch.randn(1, 2, 1024, 1024 * 1024, dtype=dtype, device=device)
        conv1(input_large)
        conv2 = torch.nn.Conv2d(1, 1024, 1, 1).to(device).to(dtype)
        input_large = torch.randn(1, 1, 2048, 1024 , dtype=dtype, device=device)
        conv2(input_large)

    def test_conv_noncontig_weights(self, device):
        for dim in (1, 2, 3):
            for grouped in (False, True):
                nc = 3
                groups = 3 if grouped else 1
                w = torch.randn([3] * dim, device=device)
                w = w.expand([nc, int(nc / groups)] + list(w.shape))
                w = w.detach().requires_grad_()
                x = torch.randn([1, nc] + ([5] * dim), device=device, requires_grad=True)
                y = getattr(F, 'conv{}d'.format(dim))(x, w, groups=groups)
                y.sum().backward()
                y = getattr(F, 'conv_transpose{}d'.format(dim))(x, w, groups=groups)
                y.sum().backward()

    def test_conv_noncontig_weights_and_bias(self, device):
        # need floats to exercise https://github.com/pytorch/pytorch/issues/16018
        for bias in [True, False]:
            conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                              bias=bias).to(device, torch.float)

            input_nc = torch.randn((1, 3, 224, 224, 2), device=device, dtype=torch.float)[:, :, :, :, 1]
            input_c = input_nc.contiguous()

            weight_nc = torch.randn((64, 3, 7, 7, 2), device=device, dtype=torch.float)[:, :, :, :, 1]
            conv1.weight = nn.Parameter(weight_nc)
            weight_c = conv1.weight.contiguous()

            if bias:
                bias_nc = torch.randn((64, 2), device=device, dtype=torch.float)[:, 1]
                conv1.bias = nn.Parameter(bias_nc)
                bias_c = conv1.bias.contiguous()

            out1 = conv1(input_nc)
            conv1.weight = nn.Parameter(weight_c)
            if bias:
                conv1.bias = nn.Parameter(bias_c)
            out2 = conv1(input_c)
            self.assertEqual(out1, out2)

    def test_save_lstm_compatibility(self, device):
        # Test that saving an LSTM in PyTorch 1.7 and older can still be
        # loaded in newer versions of PyTorch.
        model = nn.LSTM(2, 3)
        x = torch.randn(32, 5, 2)
        expected = model(x)

        # Get a state dict for PyTorch 1.7 LSTM. Before PyTorch 1.8, proj_size
        # didn't exist.
        assert model.proj_size == 0
        state_dict = model.__dict__
        del state_dict['proj_size']

        # load a model
        loaded_model = nn.LSTM(2, 3)
        loaded_model.__setstate__(state_dict)
        result = loaded_model(x)
        self.assertEqual(result, expected)

    @onlyCUDA
    @tf32_on_and_off(0.005)
    def test_grid_sample_large(self, device):
        def issue_35202():
            input_tensor = torch.rand(1, 1, 480, 640, dtype=torch.float, device=device, requires_grad=True)
            coords = torch.tensor([[-10059144, 67680944], [67680944, 67680944]], dtype=torch.float, device=device)
            coords = coords.unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 1)
            result = torch.nn.functional.grid_sample(input_tensor, coords)
            self.assertEqual(result, torch.tensor([[[[0., 0.]]]], dtype=torch.float, device=device))
            result.backward(torch.ones_like(result))
            torch.cuda.synchronize()
        issue_35202()

        def issue_24823_1(dtype):
            image = torch.arange(27, 0, -1, dtype=dtype, device=device).view(1, 1, 3, 3, 3)
            image.requires_grad_()
            grid = torch.nn.functional.affine_grid(
                torch.tensor([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]], dtype=dtype, device=device),
                (1, 1, 3, 3, 3))
            grid[:, 1, 1, 1, 0] = float('inf')
            result = torch.nn.functional.grid_sample(image, grid, padding_mode='zeros')
            self.assertEqual(result, torch.tensor([[[[[27., 26., 25.], [24., 23., 22.], [21., 20., 19.]],
                                                     [[18., 17., 16.], [15., 0., 13.], [12., 11., 10.]],
                                                     [[9., 8., 7.], [6., 5., 4.], [3., 2., 1.]]]]],
                                                  device=device, dtype=dtype))
            result.backward(torch.ones_like(result))
            expected_grad = torch.ones_like(image)
            expected_grad[0, 0, 1, 1, 1] = 0
            self.assertEqual(image.grad, expected_grad, atol=0.005, rtol=0)
        issue_24823_1(torch.half)
        issue_24823_1(torch.float)
        issue_24823_1(torch.double)

        def issue_24823_2():
            param = torch.tensor([[[-1.0e+20, 0.0, 0.0], [0.0, -1.0e+20, 0.0]]], dtype=torch.float, device=device)
            img = torch.zeros((1, 1, 4, 4), dtype=torch.float, device=device, requires_grad=True)
            grid = torch.nn.functional.affine_grid(param, img.size())
            result = torch.nn.functional.grid_sample(img, grid)
            self.assertEqual(result, torch.zeros(1, 1, 4, 4, device=device, dtype=torch.float))
            result.backward(torch.ones_like(result))
            torch.cuda.synchronize()
        issue_24823_2()

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

    @onlyCUDA
    @largeTensorTest('12GB')
    def test_conv_transposed_large(self, device):
        dtype = torch.half if self.device_type == 'cuda' else torch.float
        conv = nn.ConvTranspose2d(1, 1, 1, 1, bias=False).to(device).to(dtype)
        input_large = torch.randn(4096, 1, 512, 1024, dtype=dtype, device=device)
        # forward
        ret = conv(input_large)
        maxdiff0 = (ret.narrow(0, 0, 1024) - conv(input_large.narrow(0, 0, 1024))).abs_().max().item()
        maxdiff1 = (ret.narrow(0, 1024, 1024) - conv(input_large.narrow(0, 1024, 1024))).abs_().max().item()
        maxdiff2 = (ret.narrow(0, 2048, 1024) - conv(input_large.narrow(0, 2048, 1024))).abs_().max().item()
        maxdiff3 = (ret.narrow(0, 3072, 1024) - conv(input_large.narrow(0, 3072, 1024))).abs_().max().item()
        self.assertEqual(maxdiff0, 0)
        self.assertEqual(maxdiff1, 0)
        self.assertEqual(maxdiff2, 0)
        self.assertEqual(maxdiff3, 0)

    @onlyCUDA
    @skipCUDAIfRocm
    @largeTensorTest('12GB')
    def test_conv_large(self, device):
        dtype = torch.half if self.device_type == 'cuda' else torch.float
        conv = nn.Conv2d(2, 2, 8, 8, bias=False).to(device).to(dtype)
        input_large = torch.randn(4097, 2, 512, 512, dtype=dtype, device=device)
        # forward
        ret = conv(input_large)
        self.assertEqual(ret[:2048], conv(input_large[:2048]))
        self.assertEqual(ret[2048:4096], conv(input_large[2048:4096]))
        self.assertEqual(ret[4096:], conv(input_large[4096:]))

        # backward
        conv.zero_grad()
        # When computing the backward, we are using the `max(dim=1)`` to create
        # some sparsity. Without this sparsity, the rounding error would be
        # too large (as large as 1e-5) to satisfy the creterion (1e-6) of `assertEqual`
        ret.view(4097, -1).max(dim=1).values.sum().backward()
        del ret
        grad1 = conv.weight.grad.detach().clone()
        conv.zero_grad()
        conv(input_large[:2048]).view(2048, -1).max(dim=1).values.sum().backward()
        conv(input_large[2048:4096]).view(2048, -1).max(dim=1).values.sum().backward()
        conv(input_large[4096:]).view(1, -1).max(dim=1).values.sum().backward()
        grad2 = conv.weight.grad.detach().clone()
        # gradients are at the order of hundreds, we need to scale it to
        # the order of one so that we can compare
        scale = 1 / grad1.abs().mean()
        grad1 = grad1 * scale
        grad2 = grad2 * scale
        self.assertEqual(grad1, grad2)

    def _test_gumbel_softmax_st_shapes(self, device, dtype, shape, dim, count_expected):
        logits = torch.randn(shape, dtype=torch.float, device=device)
        logits = logits.to(dtype)

        y_draw = F.gumbel_softmax(logits, hard=True, dim=dim)

        # All values positive
        self.assertGreaterEqual(y_draw.min(), 0)
        # Shape unchanged
        self.assertTrue(y_draw.shape == logits.shape)
        # One choice per draw
        self.assertEqual(y_draw.sum(), count_expected, atol=torch.finfo(y_draw.dtype).eps, rtol=0)

    def _test_gumbel_softmax_straight_through(self, device, dtype):
        num_draws = 100

        logits = torch.tensor([[0.2, 0.8, 0.1]], device=device)
        logits = logits.reshape([1, 3])
        logits = logits.to(dtype).requires_grad_()
        probs = logits.softmax(dim=-1)

        counts = torch.zeros_like(logits)
        for _ in range(num_draws):
            y_draw = F.gumbel_softmax(logits, hard=True)
            counts = counts + y_draw

        # All values positive
        self.assertGreaterEqual(y_draw.min(), 0)
        # Each experiment should result in 1 draw.
        self.assertEqual(counts.sum(), num_draws, atol=torch.finfo(counts.dtype).eps, rtol=0)

        # check results is asymptotically as expected.
        expected = probs * num_draws
        # ~z is approximately N(0,1) for unbiased count
        z = (counts - expected) / (expected * (1 - probs)).sqrt()
        # A (lazy) approximate 99% two-sided test:
        # occurs with prob alpha~>=0.01 if unbiased
        self.assertLess(z.abs().max().item(), 2.58)

    def _test_gumbel_softmax_grad(self, device, dtype):
        # "hard" and "not hard" should propagate same gradient.
        logits_soft = torch.zeros(10, 10, dtype=dtype, device=device, requires_grad=True)
        logits_hard = torch.zeros(10, 10, dtype=dtype, device=device, requires_grad=True)

        seed = torch.random.get_rng_state()
        y_soft = F.gumbel_softmax(logits_soft, hard=False)
        torch.random.set_rng_state(seed)
        y_hard = F.gumbel_softmax(logits_hard, hard=True)

        y_soft.sum().backward()
        y_hard.sum().backward()

        # 2eps = 1x addition + 1x subtraction.
        tol = 2 * torch.finfo(dtype).eps
        self.assertEqual(logits_soft.grad, logits_hard.grad, atol=tol, rtol=0)

    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float, torch.double)
    def test_gumbel_softmax(self, device, dtype):
        self._test_gumbel_softmax_st_shapes(device, dtype, shape=[5], dim=0, count_expected=1)
        self._test_gumbel_softmax_st_shapes(device, dtype, shape=[5], dim=-1, count_expected=1)
        self._test_gumbel_softmax_st_shapes(device, dtype, shape=[5, 4], dim=1, count_expected=5)
        self._test_gumbel_softmax_st_shapes(device, dtype, shape=[5, 4, 3], dim=1, count_expected=5 * 3)
        self._test_gumbel_softmax_st_shapes(device, dtype, shape=[5, 4, 3], dim=-1, count_expected=5 * 4)
        self._test_gumbel_softmax_straight_through(device, dtype)
        self._test_gumbel_softmax_grad(device, dtype)

    def _test_rnn_retain_variables(self, device, dtype):
        rnns = [nn.LSTM(10, 20, num_layers=2).to(device, dtype),
                nn.GRU(10, 20, num_layers=2).to(device, dtype),
                nn.RNN(10, 20, num_layers=2).to(device, dtype)]
        for rnn in rnns:
            input = torch.randn(5, 6, 10, device=device, dtype=dtype, requires_grad=True)
            output = rnn(input)
            output[0].sum().backward(retain_graph=True)
            grads = [input.grad.data.clone()] + [p.grad.data.clone() for p in rnn.parameters()]
            for _ in range(4):
                rnn.zero_grad()
                input.grad.data.zero_()
                output[0].sum().backward(retain_graph=True)
                grads2 = [input.grad.data] + [p.grad.data for p in rnn.parameters()]
                self.assertEqual(grads, grads2)

    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.double)
    def test_rnn_retain_variables(self, device, dtype):
        self._test_rnn_retain_variables(device, dtype)

        if self.device_type == 'cuda' and self.has_cudnn():
            with torch.backends.cudnn.flags(enabled=False):
                self._test_rnn_retain_variables(device, dtype)

    @onlyCUDA
    def test_upsamplingNearest1d_launch_config(self, device):
        m = nn.Upsample(scale_factor=2)
        inp = torch.rand(2**25, 1, 1, device=device)
        out = m(inp)
        inp_ref = inp.cpu()
        out_ref = m(inp_ref)
        self.assertEqual(out_ref, out)

    @onlyCUDA
    def test_upsamplingNearest2d_launch_config(self, device):
        m = nn.Upsample(scale_factor=2)
        inp = torch.rand(2**25, 1, 1, 1, device=device)
        out = m(inp)
        inp_ref = inp.cpu()
        out_ref = m(inp_ref)
        self.assertEqual(out_ref, out)

    @onlyCUDA
    def test_upsamplingNearest3d_launch_config(self, device):
        m = nn.Upsample(scale_factor=2)
        inp = torch.rand(2**25, 1, 1, 1, 1, device=device)
        out = m(inp)
        inp_ref = inp.cpu()
        out_ref = m(inp_ref)
        self.assertEqual(out_ref, out)

    @unittest.expectedFailure
    @skipIfRocm
    @onlyCUDA
    def test_upsamplingNearest2d_launch_fail(self, device):
        m = nn.Upsample(scale_factor=2)
        # launch grid_y == 2**16 (larger than maximum y-dimension limit 65535)
        inp = torch.rand(1, 1, 2**15, 2**8, device=device)
        out = m(inp)

    @onlyCUDA
    @skipCUDAIfNotRocm
    def test_upsamplingNearest2d_launch_rocm(self, device):
        # test_upsamplingNearest2d_launch_fail should run OK on ROCm
        m = nn.Upsample(scale_factor=2)
        inp = torch.rand(1, 1, 2**15, 2**8, device=device)
        out = m(inp)

    @onlyCUDA
    @skipCUDAIfCudnnVersionLessThan(7600)
    def test_CTCLoss_cudnn(self, device):
        def _helper(zero_infinity):
            target_lengths = [30, 25, 20]
            input_lengths = [50, 50, 50]
            targets = torch.randint(1, 15, (sum(target_lengths),), dtype=torch.int)
            log_probs = torch.randn(50, 3, 15, dtype=torch.float, device=device).log_softmax(2).requires_grad_()

            log_probs_ref = log_probs.detach().clone().requires_grad_()

            with torch.backends.cudnn.flags(enabled=True):
                res = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths, zero_infinity=zero_infinity)
                res.backward()

            expected = ctcloss_reference(log_probs, targets.cuda(), input_lengths, target_lengths).float()

            with torch.backends.cudnn.flags(enabled=False):
                res2 = torch.nn.functional.ctc_loss(log_probs_ref, targets.cuda().long(), input_lengths, target_lengths,
                                                    zero_infinity=zero_infinity)
                res2.backward()

            self.assertEqual(res, expected)
            self.assertEqual(res2, res)
            self.assertEqual(log_probs.grad, log_probs_ref.grad)

        _helper(zero_infinity=True)
        _helper(zero_infinity=False)

    @onlyCUDA
    @skipCUDAIfNoCudnn
    def test_contig_wrong_stride_cudnn(self, device):
        # x has to have batch_size 1 to test contiguous checks
        x = torch.randn(1, 16, 5, 5, device=device)
        stride = list(x.stride())
        stride[0] = 20
        # change the stride in dimension 0. the tensor is still contiguous because size[0] is 1
        x.set_(x.storage(), 0, x.size(), stride)
        self.assertTrue(x.is_contiguous())
        F.conv_transpose2d(x, torch.randn(16, 1, 1, 1, device=device))
        F.conv2d(x, torch.randn(1, 16, 1, 1, device=device))

    @onlyCUDA
    def test_Conv2d_size_1_kernel(self, device):
        x_cpu = torch.randn(2, 3, 5, 5)
        conv_cpu = torch.nn.Conv2d(3, 3, kernel_size=1)
        y_cpu = conv_cpu(x_cpu)
        y = torch.rand_like(y_cpu)
        y_cpu.backward(y)

        with cudnn.flags(enabled=False):
            conv_cuda = torch.nn.Conv2d(3, 3, kernel_size=1).to(device)
            conv_cuda.bias.data.copy_(conv_cpu.bias.data)
            conv_cuda.weight.data.copy_(conv_cpu.weight.data)
            y_cuda = conv_cuda(x_cpu.to(device))
            y_cuda.backward(y.to(device))

        self.assertEqual(y_cpu, y_cuda, atol=1e-5, rtol=0, exact_device=False)
        self.assertEqual(conv_cpu.bias.grad.data, conv_cuda.bias.grad.data, atol=1e-5, rtol=0, exact_device=False)
        self.assertEqual(conv_cpu.weight.grad.data, conv_cuda.weight.grad.data, atol=1e-5, rtol=0, exact_device=False)

    @onlyCUDA
    def test_ConvTranspose2d_size_1_kernel(self, device):
        x_cpu = torch.randn(2, 3, 5, 5)
        conv_cpu = torch.nn.ConvTranspose2d(3, 3, kernel_size=1)
        y_cpu = conv_cpu(x_cpu)
        y = torch.rand_like(y_cpu)
        y_cpu.backward(y)

        with cudnn.flags(enabled=False):
            conv_cuda = torch.nn.ConvTranspose2d(3, 3, kernel_size=1).to(device)
            conv_cuda.bias.data.copy_(conv_cpu.bias.data)
            conv_cuda.weight.data.copy_(conv_cpu.weight.data)
            y_cuda = conv_cuda(x_cpu.to(device))
            y_cuda.backward(y.to(device))

        self.assertEqual(y_cpu, y_cuda, atol=1e-5, rtol=0, exact_device=False)
        self.assertEqual(conv_cpu.bias.grad.data, conv_cuda.bias.grad.data, atol=1e-5, rtol=0, exact_device=False)
        self.assertEqual(conv_cpu.weight.grad.data, conv_cuda.weight.grad.data, atol=1e-5, rtol=0, exact_device=False)

    @onlyCUDA
    def test_ConvTranspose3d_size_1_kernel(self, device):
        x_cpu = torch.randn(2, 3, 3, 5, 5)
        conv_cpu = torch.nn.ConvTranspose3d(3, 3, kernel_size=1)
        y_cpu = conv_cpu(x_cpu)
        y = torch.rand_like(y_cpu)
        y_cpu.backward(y)

        with cudnn.flags(enabled=False):
            conv_cuda = torch.nn.ConvTranspose3d(3, 3, kernel_size=1).to(device)
            conv_cuda.bias.data.copy_(conv_cpu.bias.data)
            conv_cuda.weight.data.copy_(conv_cpu.weight.data)
            y_cuda = conv_cuda(x_cpu.to(device))
            y_cuda.backward(y.to(device))

        self.assertEqual(y_cpu, y_cuda, atol=1e-5, rtol=0, exact_device=False)
        self.assertEqual(conv_cpu.bias.grad.data, conv_cuda.bias.grad.data, atol=1e-5, rtol=0, exact_device=False)
        self.assertEqual(conv_cpu.weight.grad.data, conv_cuda.weight.grad.data, atol=1e-5, rtol=0, exact_device=False)

    def _ordered_sequence(self, device, dtype):
        """Create ordered list of random sequences"""
        seqs = [torch.empty(random.randint(1, 6), device=device, dtype=dtype)
                for _ in range(5)]
        seqs = [s.random_(-128, 128) for s in seqs]
        ordered = sorted(seqs, key=len, reverse=True)
        return ordered

    def _padded_sequence(self, device, dtype):
        """Create Tensor of random padded sequences"""
        ordered = self._ordered_sequence(device, dtype)
        lengths = [len(i) for i in ordered]
        padded_tensor = rnn_utils.pad_sequence(ordered)
        return padded_tensor, lengths

    @onlyCUDA
    def test_device_mask(self, device):
        for enforce_sorted in [True, False]:
            padded, lengths = self._padded_sequence('cpu', torch.float)
            packed = rnn_utils.pack_padded_sequence(
                padded, lengths, enforce_sorted=enforce_sorted)
            self.assertFalse(packed.is_cuda)
            packed = packed.to(device)
            self.assertTrue(packed.is_cuda)
            unpacked, _ = rnn_utils.pad_packed_sequence(packed)
            self.assertTrue(unpacked.is_cuda)
            self.assertEqual(unpacked.dtype, torch.float)

    @onlyCUDA
    def test_overwrite_module_params_on_conversion_cpu_device(self, device):
        # Test that under the current default settings
        # (`torch.__future__.get_overwrite_module_params_on_conversion() == False`),
        # a view to a module's parameters is not pointing to the same storage as
        # its base variable after converting the module to a different device.
        m = nn.Linear(20, 10)
        mw = m.weight[:]
        m.to(device)
        with torch.no_grad():
            # Without using `torch.no_grad()`, this will leak CUDA memory.
            # (Issue is filed at https://github.com/pytorch/pytorch/issues/21875)
            mw[0][0] = 5
            self.assertTrue(mw[0][0].device.type == "cpu")
            self.assertTrue(mw._base[0][0].device.type == "cuda")

        try:
            torch.__future__.set_overwrite_module_params_on_conversion(True)

            # Test that if `torch.__future__.get_overwrite_module_params_on_conversion() == True`,
            # a view to a module's parameters is still pointing to the same storage as
            # its base variable after converting the module to a different device.
            m = nn.Linear(20, 10)
            mw = m.weight[:]
            m.to(device)
            with torch.no_grad():
                mw[0][0] = 5
            self.assertTrue(mw[0][0] == mw._base[0][0])

            # Test that if `torch.__future__.get_overwrite_module_params_on_conversion() == True`,
            # `cpu_module.to("cuda")` doesn't preserve previous references to
            # `cpu_module`'s parameters or gradients.
            m = nn.Linear(20, 10)
            m.weight.grad = torch.randn(10, 20)
            weight_ref = m.weight
            weight_grad_ref = m.weight.grad
            m.to(device)
            self.assertNotEqual(weight_ref.device, m.weight.device)
            self.assertNotEqual(weight_grad_ref.device, m.weight.grad.device)
        finally:
            torch.__future__.set_overwrite_module_params_on_conversion(False)

    @onlyCUDA
    @dtypes(*ALL_TENSORTYPES2)
    def test_embedding_max_norm_device(self, device, dtype):
        embedding = nn.Embedding(22, 5, max_norm=1.0).to(device, dtype=dtype)
        # nn.Embedding only takes LongTensor as input
        input = torch.tensor([2, 8, 8, 6], device=device, dtype=torch.long)
        output = embedding(input)
        self.assertEqual(output[1], output[2])
        self.assertTrue(output.data.norm(p=2, dim=1).le(1).all())

    # Test fails on Vg20
    @skipCUDAIfRocm
    @onlyCUDA
    @dtypes(torch.half, torch.float)
    def test_softmax(self, device, dtype):
        input = torch.rand(32, 100, device=device, dtype=dtype, requires_grad=True)
        inputf = input.to(torch.float).detach().requires_grad_(True)
        out = F.softmax(input, dim=-1, dtype=torch.float)
        outf = F.softmax(inputf, dim=-1)
        # should be bitwise equal
        self.assertEqual(out, outf, atol=0, rtol=0)
        gO = torch.empty_like(outf).uniform_()
        out.backward(gO)
        outf.backward(gO)
        # should be bitwise equal
        self.assertEqual(input.grad, inputf.grad.to(dtype), atol=0, rtol=0)

    @onlyCUDA
    def test_pool3d_size_one_feature_dim(self, device):
        # Tests crazy strides for feature dim of size 1
        x = torch.randn(7, 1, 5, 3, 2, device=device)
        strange_strides = [30, 1234, 6, 2, 1]
        y = x.as_strided(x.size(), strange_strides)
        x = x.cpu().as_strided(x.size(), strange_strides)

        to_test = {
            'max_pool3d': lambda t: F.max_pool3d(t, (5, 1, 1), stride=(5, 1, 1)),
            'avg_pool3d': lambda t: F.avg_pool3d(t, (5, 1, 1), stride=(5, 1, 1)),
        }

        for test, fn in to_test.items():
            # Should not crash
            out_y = fn(y)
            out_x = fn(x)
            self.assertEqual(out_y, out_x.to(device), msg=test)

    @onlyCUDA
    @largeTensorTest('6GB')
    def test_pool3d_large_size_int64(self, device):
        # See https://github.com/pytorch/pytorch/issues/52822
        x = torch.randn(70, 32, 100, 100, 100, dtype=torch.half, device=device)
        y = torch.nn.functional.max_pool3d(x, 5)
        torch.cuda.synchronize()

        ref_x = x.cpu().float()  # max_pool3d_cpu is not implemented for half
        ref_y = torch.nn.functional.max_pool3d(ref_x, 5)

        self.assertEqual(y, ref_y, exact_dtype=False)

    @onlyCUDA
    def test_AvgPool3d_backward_after_cat_dim1_device(self, device):
        # x has to have batch_size 1 to test contiguous checks
        x = torch.randn(1, 3, 4, 4, 4, device=device, requires_grad=True)
        y = F.avg_pool3d(x, kernel_size=3, padding=1, stride=2)

        grad = torch.randn(y.size(), device=device)
        # increase the stride in dimension 0. the tensor is still contiguous because size[0] is 1
        stride = list(grad.stride())
        stride[0] = stride[0] * 2
        grad.set_(grad.storage(), 0, grad.size(), stride)
        assert grad.is_contiguous()

        y.backward(grad)

    def test_pooling_size_empty(self, device):
        t = torch.rand([1, 2, 3, 4], device=device)
        self.assertRaises(RuntimeError, lambda: F.adaptive_avg_pool1d(t, []))
        self.assertRaises(RuntimeError, lambda: F.adaptive_avg_pool2d(t, []))
        self.assertRaises(RuntimeError, lambda: F.adaptive_avg_pool3d(t, []))
        self.assertRaises(RuntimeError, lambda: F.adaptive_max_pool1d(t, []))
        self.assertRaises(RuntimeError, lambda: F.adaptive_max_pool2d(t, []))
        self.assertRaises(RuntimeError, lambda: F.adaptive_max_pool3d(t, []))

    @dtypes(*itertools.product((torch.int, torch.long), (torch.int, torch.long)))
    def test_embedding_bag_empty_input(self, device, dtypes):
        m = 4
        n = 3
        x = torch.tensor([], device=device, dtype=dtypes[0])
        for sparse in [True, False]:
            Embed = torch.nn.EmbeddingBag(m, n, sparse=sparse)
            Embed.to(device)

            output = Embed(input=x, offsets=torch.tensor([0], device=device, dtype=dtypes[1]))
            self.assertEqual(output, torch.zeros_like(output))

            output = Embed(input=x, offsets=torch.tensor([0, 0], device=device, dtype=dtypes[1]))
            self.assertEqual(output, torch.zeros_like(output))

    @dtypes(*itertools.product((torch.int, torch.long), (torch.int, torch.long)))
    def test_EmbeddingBag_per_sample_weights_failures(self, device, dtypes):
        # Failure 1: mismatched embeddings / per_sample_weights dtype
        es = nn.EmbeddingBag(5, 2, mode='sum').to(dtype=torch.float, device=device)
        input = torch.tensor([3, 1, 1, 1, 4, 0], dtype=dtypes[0], device=device)
        offsets = torch.tensor([0, 0, 3, 3, 6], dtype=dtypes[1], device=device)
        per_sample_weights = torch.randn_like(input, dtype=torch.double, device=device)
        if device == 'cpu':
            with self.assertRaisesRegex(RuntimeError, 'have the same type as'):
                es(input, offsets, per_sample_weights)
        else:
            with self.assertRaisesRegex(RuntimeError, 'expected scalar type'):
                es(input, offsets, per_sample_weights)

        # Failure 2.1: input/per_sample_weights have different sizes (1d input)
        input = torch.tensor([3, 1, 1, 1, 4, 0], dtype=dtypes[0], device=device)
        offsets = torch.tensor([0, 0, 3, 3, 6], dtype=dtypes[1], device=device)
        per_sample_weights = torch.randn(5, dtype=torch.float, device=device)
        with self.assertRaisesRegex(ValueError, 'same shape as the input'):
            es(input, offsets, per_sample_weights)

        # Failure 2.2: input/per_sample_weights have different sizes (2d input)
        input = torch.randint(5, (7, 3), dtype=dtypes[0], device=device)
        offsets = None
        per_sample_weights = torch.randn(7 * 3, dtype=torch.float, device=device)
        with self.assertRaisesRegex(ValueError, 'same shape as the input'):
            es(input, offsets, per_sample_weights)

        # Failure 3: Unsupported per_sample_weights and mode=('max', 'mean')
        for unsupported_mode in ('max', 'mean'):
            es = nn.EmbeddingBag(5, 2, mode=unsupported_mode).to(
                dtype=torch.float, device=device)
            input = torch.randint(5, (7, 3), dtype=dtypes[0], device=device)
            offsets = None
            per_sample_weights = torch.randn(7, 3, dtype=torch.float, device=device)
            with self.assertRaisesRegex(NotImplementedError,
                                        "only supported for mode='sum'"):
                es(input, offsets, per_sample_weights)

    def _embedding_bag_reference_impl(self, input, weight, offsets=None, mode='sum',
                                      per_sample_weights=None, include_last_offset=False):
        assert mode == 'sum' or per_sample_weights is None
        assert offsets is not None
        if per_sample_weights is None:
            per_sample_weights = torch.ones(input.size()).to(
                dtype=weight.dtype, device=weight.device
            )
        assert input.numel() == per_sample_weights.numel()

        bags = []
        long_input = input.to(torch.long)
        embeddings = weight.index_select(0, long_input) * per_sample_weights.unsqueeze(1)
        if include_last_offset:
            for index in range(len(offsets) - 1):
                offset = offsets[index]
                next_offset = offsets[index + 1]
                length = next_offset - offset
                if length == 0:
                    bags.append(
                        torch.tensor([0] * weight.size(1)).to(
                            dtype=embeddings.dtype, device=embeddings.device
                        )
                    )
                else:
                    if mode == 'sum':
                        bags.append(embeddings.narrow(0, offset, length).sum(0))
                    elif mode == 'mean':
                        bags.append(embeddings.narrow(0, offset, length).sum(0).div(length))
                    else:
                        assert mode == 'max'
                        bags.append(embeddings.narrow(0, offset, length).max(0)[0])
        else:
            for index, offset in enumerate(offsets):
                if index + 1 < len(offsets):
                    next_offset = offsets[index + 1]
                else:
                    next_offset = len(long_input)
                length = next_offset - offset
                if length == 0:
                    bags.append(
                        torch.tensor([0] * weight.size(1)).to(
                            dtype=embeddings.dtype, device=embeddings.device
                        )
                    )
                else:
                    if mode == 'sum':
                        bags.append(embeddings.narrow(0, offset, length).sum(0))
                    elif mode == 'mean':
                        bags.append(embeddings.narrow(0, offset, length).sum(0).div(length))
                    else:
                        assert mode == 'max'
                        bags.append(embeddings.narrow(0, offset, length).max(0)[0])
        return torch.stack(bags)

    @dtypesIfCUDA(*itertools.product((torch.int, torch.long), (torch.int, torch.long), (torch.float, torch.double, torch.half)))
    @dtypes(*itertools.product((torch.int, torch.long), (torch.int, torch.long), (torch.float, torch.double)))
    def test_EmbeddingBag_empty_per_sample_weights_and_offsets(self, device, dtypes):
        # Test empty input and per sample weight, and backward pass. There was a CUDA
        # invalid configuration bug (more context in #46572)
        def test_per_sample_weights(mode, trainable_scale):
            es = nn.EmbeddingBag(5, 2, mode=mode).to(dtype=dtypes[2], device=device)
            es.weight.data.copy_(
                torch.arange(1, 11, device=device, dtype=dtypes[2]).view_as(es.weight))
            input = torch.tensor([], device=device, dtype=dtypes[0])
            offsets = torch.tensor([0, 0, 0, 0, 0], device=device, dtype=dtypes[1])
            per_sample_weights = torch.randn_like(input, dtype=dtypes[2]) \
                                      .requires_grad_(trainable_scale)
            ref_per_sample_weights = \
                per_sample_weights.detach().requires_grad_(trainable_scale)
            reference_weights = es.weight.detach().requires_grad_()

            expected = self._embedding_bag_reference_impl(
                input, reference_weights, offsets, mode, ref_per_sample_weights)
            result = es(input, offsets, per_sample_weights)
            self.assertEqual(result, expected, atol=dtype2prec_DONTUSE[dtypes[2]], rtol=0)

            grad = torch.randn_like(expected)
            result.backward(grad)
            # the reference impl doesn't have grad fn for empty input; but the grad should
            # simply be a zero tensor
            ref_weights_grad = torch.zeros_like(es.weight)
            self.assertEqual(es.weight.grad, ref_weights_grad,
                             atol=dtype2prec_DONTUSE[dtypes[2]], rtol=0)
            if trainable_scale:
                ref_per_sample_weights_grad = torch.empty_like(per_sample_weights)
                self.assertEqual(per_sample_weights.grad, ref_per_sample_weights_grad,
                                 atol=dtype2prec_DONTUSE[dtypes[2]], rtol=0)

        modes = ('sum',)
        trainable_scale = (True, False)
        for mode, trainable in itertools.product(modes, trainable_scale):
            test_per_sample_weights(mode, trainable)

    @dtypesIfCUDA(*itertools.product((torch.int, torch.long), (torch.int, torch.long), (torch.float, torch.double, torch.half)))
    @dtypes(*itertools.product((torch.int, torch.long), (torch.int, torch.long), (torch.float, torch.double)))
    def test_EmbeddingBag_per_sample_weights_and_offsets(self, device, dtypes):
        def test_per_sample_weights(mode, trainable_scale):
            es = nn.EmbeddingBag(5, 2, mode=mode).to(dtype=dtypes[2], device=device)
            es.weight.data.copy_(
                torch.arange(1, 11, device=device, dtype=dtypes[2]).view_as(es.weight))
            input = torch.tensor([3, 1, 1, 1, 4, 0], device=device, dtype=dtypes[0])
            offsets = torch.tensor([0, 0, 3, 3, 6], device=device, dtype=dtypes[1])
            per_sample_weights = torch.randn_like(input, dtype=dtypes[2]) \
                                      .requires_grad_(trainable_scale)
            ref_per_sample_weights = \
                per_sample_weights.detach().requires_grad_(trainable_scale)
            reference_weights = es.weight.detach().requires_grad_()

            expected = self._embedding_bag_reference_impl(
                input, reference_weights, offsets, mode, ref_per_sample_weights)
            result = es(input, offsets, per_sample_weights)
            self.assertEqual(result, expected, atol=dtype2prec_DONTUSE[dtypes[2]], rtol=0)

            grad = torch.randn_like(expected).to(dtype=dtypes[2], device=device)
            result.backward(grad)
            expected.backward(grad)
            self.assertEqual(es.weight.grad, reference_weights.grad,
                             atol=dtype2prec_DONTUSE[dtypes[2]], rtol=0)
            if trainable_scale:
                self.assertEqual(per_sample_weights.grad, ref_per_sample_weights.grad,
                                 atol=dtype2prec_DONTUSE[dtypes[2]], rtol=0)

        modes = ('sum',)
        trainable_scale = (True, False)
        for mode, trainable in itertools.product(modes, trainable_scale):
            test_per_sample_weights(mode, trainable)

    @dtypesIfCUDA(*itertools.product((torch.int, torch.long), (torch.int, torch.long), (torch.float, torch.double, torch.half)))
    @dtypes(*itertools.product((torch.int, torch.long), (torch.int, torch.long), (torch.float, torch.double)))
    def test_EmbeddingBag_per_sample_weights_and_new_offsets(self, device, dtypes):
        def test_per_sample_weights_new_offsets(mode, trainable_scale, include_last_offset, has_weight=True):
            es = nn.EmbeddingBag(5, 2, mode=mode, include_last_offset=include_last_offset).to(dtype=dtypes[2], device=device)
            es.weight.data.copy_(
                torch.arange(1, 11, device=device, dtype=dtypes[2]).view_as(es.weight))
            input = torch.tensor([3, 1, 1, 1, 4, 0], device=device, dtype=dtypes[0])
            offsets = torch.tensor([0, 0, 3, 3, 6], device=device, dtype=dtypes[1])

            if include_last_offset:
                offsets = torch.cat((offsets, torch.tensor([input.size(0)], device=device, dtype=dtypes[1])), 0)

            if has_weight:
                per_sample_weights = torch.randn_like(input, device=device, dtype=dtypes[2]) \
                                          .requires_grad_(trainable_scale)
                ref_per_sample_weights = \
                    per_sample_weights.detach().requires_grad_(trainable_scale)
            else:
                per_sample_weights = None
                ref_per_sample_weights = None

            reference_weights = es.weight.detach().requires_grad_()

            expected = self._embedding_bag_reference_impl(
                input, reference_weights, offsets, mode, ref_per_sample_weights, include_last_offset)
            result = es(input, offsets, per_sample_weights)
            self.assertEqual(result, expected, atol=dtype2prec_DONTUSE[dtypes[2]], rtol=0)

            grad = torch.randn_like(expected)
            result.backward(grad)
            expected.backward(grad)
            self.assertEqual(es.weight.grad, reference_weights.grad,
                             atol=dtype2prec_DONTUSE[dtypes[2]], rtol=0)
            if has_weight and trainable_scale:
                self.assertEqual(per_sample_weights.grad, ref_per_sample_weights.grad,
                                 atol=dtype2prec_DONTUSE[dtypes[2]], rtol=0)

        trainable_scale = (True, False)
        include_last_offset = (True, False)
        modes = (('sum', False), ('sum', True), ('max', False), ('mean', False))
        for (mode, has_weight), trainable, include_last_offset in itertools.product(
            modes, trainable_scale, include_last_offset
        ):
            test_per_sample_weights_new_offsets(
                mode, trainable, include_last_offset, has_weight
            )

    def _test_EmbeddingBag_vs_Embedding(self, N, D, B, L, max_norm=None,
                                        mode='mean',
                                        device='cpu',
                                        wdtype=torch.float,
                                        dtype=torch.long,
                                        test_per_sample_weights=False,
                                        trainable_per_sample_weights=False,
                                        sparse=False,
                                        test_backward=True,
                                        backward_prec=None):
        es = nn.EmbeddingBag(N, D, mode=mode, sparse=sparse, max_norm=max_norm).to(device, wdtype)
        e = nn.Embedding(N, D, max_norm=max_norm).to(device, wdtype)
        e.weight.data.copy_(es.weight)
        input = torch.randint(N, (B, L), device=device, dtype=dtype)
        offsets = torch.arange(0, B, device=device, dtype=dtype).mul_(L)
        grad_output = torch.rand(B, D, device=device, dtype=wdtype)

        if test_per_sample_weights:
            # To prevent large gradients, weights should sum to 1 for each bag
            per_sample_weights = \
                torch.randn(B, L, device=device, dtype=wdtype).softmax(dim=-1)
            per_sample_weights_reference = \
                per_sample_weights.clone().requires_grad_(trainable_per_sample_weights)
            per_sample_weights.requires_grad_(trainable_per_sample_weights)
            output = es(input.view(-1), offsets, per_sample_weights.view(-1))
        else:
            output = es(input.view(-1), offsets)
            per_sample_weights = None
            per_sample_weights_reference = None

        if mode == 'sum':
            if test_per_sample_weights:
                ref_output = (e(input) * per_sample_weights_reference.unsqueeze(-1)).sum(1)
            else:
                ref_output = e(input).sum(1)
        elif mode == 'mean':
            assert not test_per_sample_weights
            ref_output = e(input).mean(1)
        elif mode == 'max':
            assert not test_per_sample_weights
            ref_output = e(input).max(1)[0]

        self.assertEqual(output, ref_output, atol=dtype2prec_DONTUSE[wdtype], rtol=0)

        if not test_backward:
            return

        output.backward(grad_output)
        ref_output.backward(grad_output)
        es_weight_grad = es.weight.grad.data
        if sparse:
            es_weight_grad = es.weight.grad.data.to_dense()

        # We have more floating point error here because we are dealing with larger numbers
        if backward_prec is None:
            needed_prec = dtype2prec_DONTUSE[wdtype] * 5
        else:
            needed_prec = backward_prec

        self.assertEqual(es_weight_grad, e.weight.grad, atol=needed_prec, rtol=0)

        if test_per_sample_weights and trainable_per_sample_weights:
            self.assertEqual(per_sample_weights.grad, per_sample_weights_reference.grad,
                             atol=dtype2prec_DONTUSE[wdtype], rtol=0)

    @skipCUDAIf(True, "Temporarily disabled. See t54369166")
    @dtypesIfCUDA(*itertools.product((torch.int, torch.long), (torch.half, torch.float, torch.double)))
    @dtypes(*itertools.product((torch.int, torch.long), (torch.float, torch.double)))
    def test_EmbeddingBag_per_sample_weights_and_no_offsets(self, device, dtypes):
        def run_tests(mode, sparse, trainable_per_sample_weights):
            kwargs = dict(test_per_sample_weights=True, device=device,
                          mode=mode, wdtype=dtypes[1], dtype=dtypes[0], sparse=sparse,
                          trainable_per_sample_weights=trainable_per_sample_weights)

            # Simple case
            self._test_EmbeddingBag_vs_Embedding(2, 3, 5, 7, **kwargs)

            # B * L > 1000
            self._test_EmbeddingBag_vs_Embedding(2, 5, 53, 23, **kwargs)

            # Large num_embedding
            self._test_EmbeddingBag_vs_Embedding(101, 5, 3, 7, **kwargs)

            # Large embedding_dim
            self._test_EmbeddingBag_vs_Embedding(2, 101, 3, 7, **kwargs)

        modes = ('sum',)
        sparsity = (True, False)
        trainable_scale = (True, False)
        for mode, sparse, trainable_per_sample_weights in \
                itertools.product(modes, sparsity, trainable_scale):
            run_tests(mode, sparse, trainable_per_sample_weights)

        # Test CUDA Dense on half precision
        if device == 'cuda':
            modes = ('sum',)
            sparsity = (False,)
            trainable_scale = (True, False)
            for mode, sparse, trainable_per_sample_weights in \
                    itertools.product(modes, sparsity, trainable_scale):
                run_tests(mode, sparse, trainable_per_sample_weights)

    def _test_EmbeddingBag(
        self,
        device,
        mode,
        sparse,
        wdtype=torch.double,
        dtype=torch.long,
        odtype=torch.long,
        test_backward=True,
    ):
        # check a known test example
        es = nn.EmbeddingBag(5, 2, mode=mode, sparse=sparse).to(device, wdtype)
        es.weight.data.copy_(torch.arange(1, 11, device=device, dtype=wdtype).view_as(es.weight))
        input = torch.tensor([3, 1, 1, 1, 4, 0], device=device, dtype=dtype)
        offsets = torch.tensor([0, 0, 3, 3, 6], device=device, dtype=odtype)

        grad_output = torch.tensor(
            [1, 2,
             3, 4], device=device, dtype=wdtype).view(2, 2)
        grad_output_with_empty = torch.tensor(
            [99, 99,
             1, 2,
             99, 99,
             3, 4,
             99, 99], device=device, dtype=wdtype).view(5, 2)

        if mode == "sum" or mode == "mean":
            denominator = 1 if mode == "sum" else 3
            expected_output = torch.tensor(
                [[13, 16],
                 [13, 16]], device=device, dtype=wdtype) / denominator

            expected_output_with_empty = torch.tensor(
                [[0, 0],
                 [13, 16],
                 [0, 0],
                 [13, 16],
                 [0, 0]], device=device, dtype=wdtype) / denominator

            expected_grad_weight = torch.tensor(
                [[3, 4],
                 [5, 8],
                 [0, 0],
                 [1, 2],
                 [3, 4]], device=device, dtype=wdtype) / denominator
        elif mode == "max":
            expected_output = torch.tensor(
                [[7, 8],
                 [9, 10]], device=device, dtype=wdtype)

            expected_output_with_empty = torch.tensor(
                [[0, 0],
                 [7, 8],
                 [0, 0],
                 [9, 10],
                 [0, 0]], device=device, dtype=wdtype)

            expected_grad_weight = torch.tensor(
                [[0, 0],
                 [0, 0],
                 [0, 0],
                 [1, 2],
                 [3, 4]], device=device, dtype=wdtype)
        output = es(input, offsets)
        output.backward(grad_output_with_empty)

        es_weight_grad = es.weight.grad.data
        if sparse:
            es_weight_grad = es.weight.grad.to_dense()
        self.assertEqual(output, expected_output_with_empty)
        self.assertEqual(es_weight_grad, expected_grad_weight, atol=dtype2prec_DONTUSE[wdtype], rtol=0)

        # check same example except as 2D (2 x 3)
        input = input.view(2, -1)
        es.zero_grad()
        output = es(input)
        output.backward(grad_output)

        es_weight_grad = es.weight.grad
        if sparse:
            es_weight_grad = es.weight.grad.to_dense()
        self.assertEqual(output, expected_output)
        self.assertEqual(es_weight_grad, expected_grad_weight, atol=dtype2prec_DONTUSE[wdtype], rtol=0)

        # test all empty bags
        es.zero_grad()
        inputs = torch.tensor([], dtype=dtype, device=device)
        offsets = torch.tensor([0, 0, 0, 0], dtype=odtype, device=device)
        es(inputs, offsets).sum().backward()
        dense_grad = es.weight.grad
        if dense_grad.is_sparse:
            dense_grad = dense_grad.to_dense()
        self.assertEqual(dense_grad, torch.zeros_like(es.weight))

        # now compare EmbeddingBag vs Embedding + Sum/Mean, for constant bag length
        N, D, B, L = random.randint(1, 100), random.randint(1, 100), random.randint(1, 50), random.randint(1, 50)
        kwargs = dict(mode=mode, sparse=sparse, device=device, wdtype=wdtype, dtype=dtype, test_backward=test_backward)
        self._test_EmbeddingBag_vs_Embedding(N, D, B, L, **kwargs)
        for max_norm in (None, 3):
            for p in itertools.product([1, 2], repeat=4):
                self._test_EmbeddingBag_vs_Embedding(*p, max_norm=max_norm, **kwargs)

        # check that giving illegal input combos raises error
        es = nn.EmbeddingBag(10, 20, mode=mode, sparse=sparse)
        input = torch.ones(3, 4, dtype=dtype)
        offset = torch.arange(0, 3, dtype=odtype)
        self.assertRaises(ValueError, lambda: es(input, offset))
        self.assertRaises(ValueError, lambda: es(input.view(-1)))
        offset[0] = 1
        if self.device_type == "cpu":
            self.assertRaises(RuntimeError, lambda: es(input.view(-1), offset))
            offset[0] = 0
            offset[-1] = 100
            self.assertRaises(RuntimeError, lambda: es(input.view(-1), offset))

    @dtypesIfCUDA(*itertools.product((torch.int, torch.long), (torch.int, torch.long), (torch.float, torch.double, torch.half)))
    @dtypes(*itertools.product((torch.int, torch.long), (torch.int, torch.long), (torch.float, torch.double)))
    def test_embedding_bag_device(self, device, dtypes):
        self._test_EmbeddingBag(device, 'sum', False, wdtype=dtypes[2], dtype=dtypes[0], odtype=dtypes[1])
        self._test_EmbeddingBag(device, 'mean', False, wdtype=dtypes[2], dtype=dtypes[0], odtype=dtypes[1])
        self._test_EmbeddingBag(device, 'max', False, wdtype=dtypes[2], dtype=dtypes[0], odtype=dtypes[1])

        test_backward = False
        if self.device_type == 'cuda':
            # see 'todo' in test_embedding_bag.
            test_backward = dtypes[2] is not torch.float16
        elif self.device_type == 'cpu':
            # TODO: figure out why precision on sparse embeddings isn't the
            # same as for dense.
            test_backward = dtypes[2] is not torch.float

        self._test_EmbeddingBag(
            device,
            'sum',
            True,
            wdtype=dtypes[2],
            dtype=dtypes[0],
            odtype=dtypes[1],
            test_backward=test_backward,
        )
        self._test_EmbeddingBag(
            device,
            'mean',
            True,
            wdtype=dtypes[2],
            dtype=dtypes[0],
            odtype=dtypes[1],
            test_backward=test_backward,
        )

    @dtypesIfCUDA(*itertools.product((torch.int, torch.long), (torch.int, torch.long), (torch.float, torch.double, torch.half)))
    @dtypes(*itertools.product((torch.int, torch.long), (torch.int, torch.long), (torch.float, torch.double)))
    def test_embedding_bag_non_contiguous_weight(self, device, dtypes):
        weight_tensor = torch.randn(3, 4, dtype=dtypes[2], device=device)

        weight_tensor_non_contig = weight_tensor[:, :3]  # This is non-contiguous strided.
        weight_tensor_contig = weight_tensor_non_contig.clone().contiguous()  # Contig-strided.

        index = torch.tensor([0, 1, 2], dtype=dtypes[0], device=device)
        offsets = torch.tensor([0, 2], dtype=dtypes[1], device=device)
        for mode in ['sum', 'mean', 'max']:
            output_non_contig = F.embedding_bag(
                input=index,
                weight=weight_tensor_non_contig,
                offsets=offsets,
                mode=mode,
            )
            output_contig = F.embedding_bag(
                input=index,
                weight=weight_tensor_contig,
                offsets=offsets,
                mode=mode,
            )
        self.assertEqual(output_non_contig, output_contig)


    @onlyCUDA
    @dtypes(*itertools.product((torch.int, torch.long), (torch.int, torch.long)))
    def test_embedding_bag_bfloat16(self, device, dtypes):
        self._test_EmbeddingBag(device, 'sum', True, wdtype=torch.bfloat16, dtype=dtypes[0], odtype=dtypes[1], test_backward=True)
        self._test_EmbeddingBag(device, 'mean', True, wdtype=torch.bfloat16, dtype=dtypes[0], odtype=dtypes[1], test_backward=True)


    @onlyCUDA
    @dtypes(torch.half, torch.float, torch.double)
    def test_multihead_attention_dtype(self, device, dtype):
        embed_dim = 128
        num_heads = 8
        sl = 10
        bs = 8
        model = nn.MultiheadAttention(embed_dim, num_heads).cuda().to(dtype)
        q = torch.randn(sl, bs, embed_dim, device=device, dtype=dtype)
        k = torch.randn(sl, bs, embed_dim, device=device, dtype=dtype)
        v = torch.randn(sl, bs, embed_dim, device=device, dtype=dtype)
        out = model(q, k, v)
        self.assertEqual(q.size(), out[0].size())
        self.assertEqual(dtype, out[0].dtype)

    @dtypesIfCUDA(*get_all_fp_dtypes(include_bfloat16=AMPERE_OR_ROCM))
    @dtypes(torch.float)
    def test_Conv2d_naive_groups(self, device, dtype):
        # Check that grouped convolutions matches two half convolutions
        m = nn.Conv2d(4, 4, kernel_size=3, groups=2).to(device, dtype)
        i = torch.randn(2, 4, 6, 6, device=device, dtype=dtype, requires_grad=True)
        output = m(i)
        grad_output = torch.randn(2, 4, 4, 4, device=device, dtype=dtype)
        output.backward(grad_output)

        m1 = nn.Conv2d(2, 2, kernel_size=3).to(device, dtype)
        m1.weight.data.copy_(m.weight.data[:2])
        m1.bias.data.copy_(m.bias.data[:2])
        i1 = i.data[:, :2].contiguous().requires_grad_(True)
        output1 = m1(i1)
        output1.backward(grad_output[:, :2].contiguous())

        m2 = nn.Conv2d(2, 2, kernel_size=3).to(device, dtype)
        m2.weight.data.copy_(m.weight.data[2:])
        m2.bias.data.copy_(m.bias.data[2:])
        i2 = i.data[:, 2:].contiguous().requires_grad_(True)
        output2 = m2(i2)
        output2.backward(grad_output[:, 2:].contiguous())

        self.assertEqual(output, torch.cat([output1, output2], 1))
        self.assertEqual(i.grad.data,
                         torch.cat([i1.grad.data, i2.grad.data], 1),
                         atol=dtype2prec_DONTUSE[dtype], rtol=0)
        self.assertEqual(m.bias.grad.data,
                         torch.cat([m1.bias.grad.data, m2.bias.grad.data], 0),
                         atol=dtype2prec_DONTUSE[dtype], rtol=0)
        self.assertEqual(m.weight.grad.data,
                         torch.cat([m1.weight.grad.data, m2.weight.grad.data], 0),
                         atol=dtype2prec_DONTUSE[dtype], rtol=0)

    def _test_batchnorm_grad(self, device, dtype=torch.double):
        bs, n_feat, size_feat = 4, 5, 6
        input = torch.arange(bs * n_feat * size_feat, device=device,
                             requires_grad=True, dtype=dtype).view(bs, n_feat, size_feat)
        weight = torch.arange(1, n_feat + 1, device=device, requires_grad=True, dtype=dtype)
        bias = torch.arange(n_feat, device=device, requires_grad=True, dtype=dtype)
        running_mean = 1 - torch.arange(n_feat, device=device, dtype=dtype)
        running_var = 2 * torch.arange(n_feat, device=device, dtype=dtype)
        for training in [False, True]:
            _assertGradAndGradgradChecks(self, F.batch_norm, (input, running_mean, running_var, weight, bias,
                                                              training, 0.1, 0.0001))

    def test_batchnorm_grad(self, device):
        self._test_batchnorm_grad(device)

        if self.device_type == 'cuda' and self.has_cudnn():
            with torch.backends.cudnn.flags(enabled=False):
                self._test_batchnorm_grad(device)


    def test_hardsigmoid_grad(self, device):
        inputs = (torch.randn(4, 16, 16, device=device) - 0.5) * 10
        inputs.requires_grad = True
        self.assertTrue(gradcheck(F.hardsigmoid, (inputs,)))

    # currently fails on XLA
    @onlyOnCPUAndCUDA
    def test_hardswish_grad(self, device):
        inputs = (torch.randn(4, 16, 16, device=device) - 0.5) * 10
        inputs.requires_grad = True
        self.assertTrue(gradcheck(F.hardswish, (inputs,)))


    def _test_batchnorm_eval(self, device, dtype, module_dtype=None):
        module_dtype = module_dtype or dtype
        module = nn.BatchNorm1d(3).to(device, module_dtype)
        module.eval()

        data = torch.rand(4, 3, device=device, dtype=dtype, requires_grad=True)
        grad = torch.rand(4, 3, device=device, dtype=dtype)

        # 1st pass
        res1 = module(data)
        res1.backward(grad)
        grad1 = data.grad.clone()

        # 2nd pass
        if data.grad is not None:
            data.grad.data.zero_()

        res2 = module(data)
        res2.backward(grad)
        grad2 = data.grad.clone()
        self.assertEqual(res1, res2)
        self.assertEqual(grad1, grad2)

        # track_running_stats=False
        module = nn.BatchNorm1d(3, track_running_stats=False).to(device, module_dtype)

        data = torch.rand(4, 3, device=device, dtype=dtype, requires_grad=True)
        grad = torch.rand(4, 3, device=device, dtype=dtype)

        # 1st pass
        res1 = module(data)
        res1.backward(grad)
        grad1 = data.grad.clone()

        # set eval
        module.eval()

        # 2nd pass
        if data.grad is not None:
            data.grad.data.zero_()

        res2 = module(data)
        res2.backward(grad)
        grad2 = data.grad.clone()
        self.assertEqual(res1, res2)
        self.assertEqual(grad1, grad2)

    @dtypes(torch.float)
    @dtypesIfCUDA(torch.float, torch.bfloat16)
    def test_batchnorm_eval(self, device, dtype):
        self._test_batchnorm_eval(device, dtype)

        if self.device_type == 'cuda' and self.has_cudnn():
            with torch.backends.cudnn.flags(enabled=False):
                self._test_batchnorm_eval(device, dtype)

    @onlyCUDA
    @dtypes(torch.bfloat16, torch.half)
    def test_batchnorm_eval_mixed(self, device, dtype):
        # Test bfloat16 input with float module
        self._test_batchnorm_eval(device, dtype, torch.float)

        if self.device_type == 'cuda' and self.has_cudnn():
            with torch.backends.cudnn.flags(enabled=False):
                self._test_batchnorm_eval(device, dtype, torch.float)

    def _test_batchnorm_simple_average(self, device, dtype, module_dtype=None):
        module_dtype = module_dtype or dtype
        module = nn.BatchNorm1d(3, momentum=None).to(dtype=module_dtype, device=device)
        zeros = torch.zeros(3, dtype=module_dtype, device=device)
        ones = torch.ones(3, dtype=module_dtype, device=device)
        self.assertEqual(module.running_mean, zeros)
        self.assertEqual(module.running_var, ones)

        data1 = torch.rand(4, 3, dtype=dtype, device=device)
        data2 = torch.rand(4, 3, dtype=dtype, device=device)

        # 1st pass
        res1 = module(data1)
        running_mean1 = module.running_mean.clone()
        running_var1 = module.running_var.clone()
        self.assertNotEqual(running_mean1, zeros)
        self.assertNotEqual(running_var1, ones)

        # reset stats
        module.reset_running_stats()
        self.assertEqual(module.running_mean, zeros)
        self.assertEqual(module.running_var, ones)

        # 2nd pass
        res2 = module(data2)
        running_mean2 = module.running_mean.clone()
        running_var2 = module.running_var.clone()
        self.assertNotEqual(running_mean2, zeros)
        self.assertNotEqual(running_var2, ones)

        # reset stats
        module.reset_running_stats()
        self.assertEqual(module.running_mean, zeros)
        self.assertEqual(module.running_var, ones)

        # 3rd (combined) pass
        res3 = module(data1)
        res4 = module(data2)
        self.assertEqual(res3, res1)
        self.assertEqual(res4, res2)
        self.assertEqual(module.running_mean, (running_mean1 + running_mean2) / 2)
        self.assertEqual(module.running_var, (running_var1 + running_var2) / 2)

    @dtypes(torch.float)
    @dtypesIfCUDA(torch.float, torch.bfloat16)
    def test_batchnorm_simple_average(self, device, dtype):
        self._test_batchnorm_simple_average(device, dtype)

        if self.device_type == 'cuda' and self.has_cudnn():
            with torch.backends.cudnn.flags(enabled=False):
                self._test_batchnorm_simple_average(device, dtype)

    @onlyCUDA
    @dtypes(torch.bfloat16, torch.half)
    def test_batchnorm_simple_average_mixed(self, device, dtype):
        self._test_batchnorm_simple_average(device, dtype, torch.float)

        if self.device_type == 'cuda' and self.has_cudnn():
            with torch.backends.cudnn.flags(enabled=False):
                self._test_batchnorm_simple_average(device, dtype, torch.float)

    def _test_maxpool_indices(self, num_dim, adaptive=False, device="cpu", dtype=torch.float):
        def expected_indices(dim):
            if dim == 1:
                return torch.tensor([1, 3], dtype=torch.double).repeat(2, 2, 1)
            if dim == 2:
                return torch.tensor([[5, 7], [13, 15]], dtype=torch.double).repeat(2, 2, 1, 1)

        def expected_grad(dim):
            if dim == 1:
                return torch.tensor([0, 1, 0, 1], dtype=torch.double).repeat(2, 2, 1)
            grad = expected_grad(dim - 1)
            zero = torch.zeros(grad.size())
            return torch.stack((zero, grad, zero, grad), 2)

        def expected_output(dim):
            if dim == 1:
                return torch.arange(2, 17, 2).view(2, 2, 2)
            if dim == 2:
                col = torch.arange(6, 63, 8)
                return torch.stack([col, col + 2], 1).view(2, 2, 2, 2)

        if adaptive:
            cls_name = 'AdaptiveMaxPool{}d'.format(num_dim)
        else:
            cls_name = 'MaxPool{}d'.format(num_dim)
        module_cls = getattr(nn, cls_name)
        module = module_cls(2, return_indices=True).to(device, dtype=dtype)
        numel = 4 ** (num_dim + 1)
        input = torch.arange(1, numel + 1).view(2, 2, *repeat(4, num_dim)).to(device, dtype=dtype)
        input_var = input.clone().detach().requires_grad_()

        # Check forward
        output, indices = module(input_var)
        if num_dim != 3:
            expected_indices = expected_indices(num_dim)
            expected_output = expected_output(num_dim)
            self.assertEqual(indices.dim(), input.dim())
            # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
            self.assertEqualIgnoreType(indices.data.squeeze(), expected_indices)
            # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
            self.assertEqualIgnoreType(output.data.squeeze(), expected_output)
        self.assertTrue(output.requires_grad)
        self.assertFalse(indices.requires_grad)

        # Make sure backward works
        grad_output = torch.ones(output.size(), device=device, dtype=dtype)
        output.backward(grad_output, retain_graph=True)
        expected_grad = expected_grad(num_dim)
        # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
        self.assertEqualIgnoreType(input_var.grad.data, expected_grad.view_as(input))

        # Make sure backward after changing indices will result in an error
        indices.add_(1)
        self.assertRaises(RuntimeError, lambda: output.backward(grad_output))

        # Make sure -Infinity is handled correctly
        t = torch.tensor([[[float("-inf")]]])
        m = nn.MaxPool1d(kernel_size=1, return_indices=True)
        output, indices = m(t)
        self.assertEqual(output[0, 0, 0], float("-inf"))
        self.assertEqual(indices[0, 0, 0], 0)

        t = torch.tensor([[[float("-inf")]]])
        m = nn.MaxPool2d(kernel_size=1, return_indices=True)
        output, indices = m(t)
        self.assertEqual(output[0, 0, 0], float("-inf"))
        self.assertEqual(indices[0, 0, 0], 0)

        t = torch.tensor([[[[float("-inf")]]]])
        m = nn.MaxPool3d(kernel_size=1, return_indices=True)
        output, indices = m(t)
        self.assertEqual(output[0, 0, 0, 0], float("-inf"))
        self.assertEqual(indices[0, 0, 0, 0], 0)

    @dtypesIfCUDA(*get_all_fp_dtypes())
    @dtypes(torch.float)
    def test_MaxPool1d_indices(self, device, dtype):
        self._test_maxpool_indices(1, device=device, dtype=dtype)

    @dtypesIfCUDA(*get_all_fp_dtypes())
    @dtypes(torch.float)
    def test_MaxPool2d_indices(self, device, dtype):
        self._test_maxpool_indices(2, device=device, dtype=dtype)

    @dtypesIfCUDA(*get_all_fp_dtypes())
    @dtypes(torch.float)
    def test_MaxPool3d_indices(self, device, dtype):
        self._test_maxpool_indices(3, device=device, dtype=dtype)

    @dtypesIfCUDA(*get_all_fp_dtypes())
    @dtypes(torch.float)
    def test_AdaptiveMaxPool1d_indices(self, device, dtype):
        self._test_maxpool_indices(1, adaptive=True, device=device, dtype=dtype)

    @dtypesIfCUDA(*get_all_fp_dtypes())
    @dtypes(torch.float)
    def test_AdaptiveMaxPool2d_indices(self, device, dtype):
        self._test_maxpool_indices(2, adaptive=True, device=device, dtype=dtype)

    @dtypesIfCUDA(*get_all_fp_dtypes())
    @dtypes(torch.float)
    def test_AdaptiveMaxPool3d_indices(self, device, dtype):
        self._test_maxpool_indices(3, adaptive=True, device=device, dtype=dtype)

    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float)
    @onlyOnCPUAndCUDA  # TODO: Fails on XLA
    def test_max_pool_nan_inf(self, device, dtype):
        for adaptive in ['', 'adaptive_']:
            for num_dim in [1, 2, 3]:
                fn_name = '{}max_pool{}d'.format(adaptive, num_dim)
                fn = getattr(F, fn_name)

                x = torch.full([1, 1] + num_dim * [3], nan, device=device, dtype=dtype, requires_grad=True)
                res = fn(x, 1 if adaptive else 3)
                res.backward(torch.randn_like(res))
                self.assertTrue(math.isnan(res.item()))
                x.requires_grad_(False)
                res = fn(x, 1 if adaptive else 3)
                self.assertTrue(math.isnan(res.item()))

                x2 = torch.full([1, 1] + num_dim * [3], -inf, device=device, dtype=dtype, requires_grad=True)
                res2 = fn(x2, 1 if adaptive else 3)
                res2.backward(torch.randn_like(res2))
                self.assertTrue(math.isinf(res2.item()))
                x2.requires_grad_(False)
                res2 = fn(x2, 1 if adaptive else 3)
                self.assertTrue(math.isinf(res2.item()))

    @onlyOnCPUAndCUDA
    @dtypes(torch.float, torch.double)
    def test_grid_sample_nan_inf(self, device, dtype):
        input = torch.zeros([1, 1, 3, 3], device=device, dtype=dtype)
        grid = torch.tensor([[[[nan, 0], [0, inf]]]], device=device, dtype=dtype)
        for padding_mode in ('reflection', 'border', 'zeros'):
            sample = torch.nn.functional.grid_sample(input=input, grid=grid, mode='nearest',
                                                     padding_mode=padding_mode, align_corners=False)
            self.assertEqual(sample, torch.zeros([1, 1, 1, 2], device=device, dtype=dtype))

    @onlyOnCPUAndCUDA
    def test_fractional_max_pool2d(self, device):
        x = torch.randn(1, 2, 7, 7, requires_grad=True, device=device)
        samples = x.new(1, 2, 2).uniform_()

        def func(x):
            return F.fractional_max_pool2d(
                x, (2, 2), output_size=(3, 3), _random_samples=samples)

        self.assertEqual(func(x).shape, (1, 2, 3, 3))
        gradcheck(func, [x])
        gradgradcheck(func, [x])

        x = torch.randn(2, 7, 7, requires_grad=True, device=device)
        self.assertEqual(func(x).shape, (2, 3, 3))
        if self.device_type != 'cuda':
            # Reference: https://github.com/pytorch/pytorch/issues/52427
            # Raises -> RuntimeError: TensorAccessor expected 4 dims but tensor has 3
            # on CUDA in gradcheck
            gradcheck(func, [x])
            gradgradcheck(func, [x])

        for kernel_size in [(), (1,)]:
            with self.assertRaisesRegex(RuntimeError, "kernel_size must either"):
                # Incorrect kernel_size
                F.fractional_max_pool2d(x, kernel_size=kernel_size, output_size=(3, 3), _random_samples=samples)

        err_large_msg = "too large relative to input "
        err_out_size_msg = "output_size must either"
        for output_size, msg in [((9, 3), err_large_msg + "height"),
                                 ((3, 9), err_large_msg + "width"),
                                 ((3,), err_out_size_msg),
                                 ((), err_out_size_msg)]:
            with self.assertRaisesRegex(RuntimeError, msg):
                # Incorrect output_size
                F.fractional_max_pool2d(x, (2, 2), output_size=output_size, _random_samples=samples)

    @onlyOnCPUAndCUDA
    def test_fractional_max_pool3d(self, device):
        x = torch.randn(1, 2, 7, 7, 7, requires_grad=True, device=device)
        samples = x.new(1, 2, 3).uniform_()

        def func(x):
            return F.fractional_max_pool3d(
                x, (2, 2, 2), output_size=(3, 3, 3), _random_samples=samples)

        self.assertEqual(func(x).shape, (1, 2, 3, 3, 3))
        gradcheck(func, [x])
        gradgradcheck(func, [x])

        x = torch.randn(2, 7, 7, 7, requires_grad=True, device=device)
        self.assertEqual(func(x).shape, (2, 3, 3, 3))
        gradcheck(func, [x])
        gradgradcheck(func, [x])

        for kernel_size in [(), (1,), (1, 1)]:
            with self.assertRaisesRegex(RuntimeError, "kernel_size must either"):
                # Incorrect kernel_size
                F.fractional_max_pool3d(x, kernel_size=kernel_size, output_size=(3, 3, 3), _random_samples=samples)

        err_large_msg = "too large relative to input "
        err_out_size_msg = "output_size must either"
        for output_size, msg in [((9, 3, 3), err_large_msg + "time"),
                                 ((3, 9, 3), err_large_msg + "height"),
                                 ((3, 3, 9), err_large_msg + "width"),
                                 ((3, 3), err_out_size_msg),
                                 ((3,), err_out_size_msg),
                                 ((), err_out_size_msg)]:
            with self.assertRaisesRegex(RuntimeError, msg):
                # Incorrect output_size
                F.fractional_max_pool3d(x, (2, 2, 2), output_size=output_size, _random_samples=samples)

    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float)
    @onlyOnCPUAndCUDA  # TODO: Fails on XLA
    def test_fractional_max_pool_nan_inf(self, device, dtype):
        for num_dim in [2, 3]:
            fn_name = 'FractionalMaxPool{}d'.format(num_dim)
            fn = getattr(nn, fn_name)(kernel_size=2, output_size=1)
            x = torch.full([1, 1] + num_dim * [3], nan, device=device, dtype=dtype, requires_grad=True)
            res = fn(x)
            res.backward(torch.randn_like(res))
            self.assertTrue(math.isnan(res.item()))

            x2 = torch.full([1, 1] + num_dim * [3], -inf, device=device, dtype=dtype, requires_grad=True)
            res2 = fn(x2)
            res2.backward(torch.randn_like(res2))
            self.assertTrue(math.isinf(res2.item()))

    @onlyOnCPUAndCUDA  # TODO: RuntimeError message different on XLA
    def test_pooling_zero_stride(self, device):
        for op in ('max', 'avg'):
            for num_dim in [1, 2, 3]:
                fn_name = '{}_pool{}d'.format(op, num_dim)
                fn = getattr(F, fn_name)
                x = torch.ones([1, 2] + num_dim * [4], device=device, dtype=torch.float)
                self.assertRaisesRegex(RuntimeError, r"stride should not be zero|stride must be greater than zero",
                                       lambda: fn(x, kernel_size=2, stride=0))

                fn_module_name = '{}Pool{}d'.format(op.title(), num_dim)
                fn_module = getattr(nn, fn_module_name)(kernel_size=2, stride=0)
                self.assertRaisesRegex(RuntimeError, r"stride should not be zero|stride must be greater than zero",
                                       lambda: fn_module(x))

    @dtypesIfCUDA(*get_all_fp_dtypes())
    @dtypes(torch.float)
    def test_pool_large_size(self, device, dtype):
        for op in ('max', 'avg'):
            for num_dim in [1, 2, 3]:
                fn_name = '{}_pool{}d'.format(op, num_dim)
                fn = getattr(F, fn_name)
                # 16777217 is the smallest integer not expressible in float32
                x = torch.ones([1, 1, 16777217] + (num_dim - 1) * [1],
                               device=device, dtype=dtype)
                res = fn(x, 1, stride=1, padding=0)
                # check if the output shape was still computed correctly
                self.assertEqual(x.shape[2], res.shape[2])

    @dtypesIfCUDA(*get_all_fp_dtypes())
    @dtypes(torch.float)
    def test_pool_invalid_size(self, device, dtype):
        for op in ('max', 'avg'):
            for num_dim in [1, 2, 3]:
                fn_name = '{}_pool{}d'.format(op, num_dim)
                if op == 'max':
                    # New implementation without indices supports empty tensors
                    # TODO(Heitor) change once with_indices code is updated
                    fn_name += '_with_indices'
                fn = getattr(F, fn_name)
                # use a configuration that gives zero outputs only
                # when doing a correct floor division by the stride
                x = torch.ones([1, 1] + num_dim * [4],
                               device=device, dtype=dtype)
                with self.assertRaisesRegex(RuntimeError, r"too small|smaller than"):
                    try:
                        res = fn(x, 3, stride=2, padding=0, dilation=2)
                    except TypeError:
                        # some implementations do not support dilation
                        res = fn(x, 6, stride=2, padding=0)

    def test_CTCLoss_empty_target(self, device):
        target_lengths = [0, 0, 0]
        input_lengths = [50, 50, 50]
        targets = torch.randint(1, 15, (0,), dtype=torch.long, device=device)
        log_probs = torch.randn(50, 3, 15, dtype=torch.double, device=device).log_softmax(2)
        loss = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths, reduction='none')
        self.assertTrue((loss >= 0).all().item())
        self.assertEqual(-log_probs.sum(0)[:, 0], loss)

        target_lengths = [0, 9, 0]
        input_lengths = [50, 50, 50]
        targets = torch.randint(1, 15, (9,), dtype=torch.long, device=device)
        log_probs = torch.randn(50, 3, 15, dtype=torch.double, device=device).log_softmax(2)
        loss = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths, reduction='none')
        self.assertTrue((loss >= 0).all().item())
        self.assertEqual(-log_probs.sum(0)[[0, 2], 0], loss[[0, 2]])

    def test_empty_dropout(self, device):
        x = torch.tensor([]).to(device)
        out = torch.nn.functional.dropout(x)
        self.assertEqual(out.size(), x.size())

    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float)
    @tf32_on_and_off(0.005)
    def test_variable_sequence(self, device, dtype):
        def pad(var, length):
            if var.size(0) == length:
                return var
            return torch.cat([var, var.new_zeros(length - var.size(0), *var.size()[1:])])

        def maybe_index_tuple(maybe_tuple_of_tensors, index):
            if maybe_tuple_of_tensors is None:
                return None
            return tuple(maybe_tuple_of_tensors[j][:, index:index + 1, :].contiguous()
                         for j in range(2))

        def check_lengths(lengths, enforce_sorted, use_default_hiddens, proj_size):
            input_size = 3
            hidden_size = 4
            num_layers = 2
            bidirectional = True

            max_length = max(lengths)
            x_leaf = torch.randn(max_length, len(lengths), input_size, device=device,
                                 dtype=dtype, requires_grad=True)
            num_directions = 2 if bidirectional else 1
            lstm = nn.LSTM(input_size, hidden_size, bidirectional=bidirectional,
                           num_layers=num_layers, proj_size=proj_size).to(device, dtype)
            lstm2 = deepcopy(lstm).to(device, dtype)
            x = x_leaf

            hidden0 = None
            if not use_default_hiddens:
                real_hidden_size = hidden_size if proj_size == 0 else proj_size
                hidden0 = (torch.randn(num_directions * num_layers, len(lengths), real_hidden_size,
                                       device=device, dtype=dtype),
                           torch.randn(num_directions * num_layers, len(lengths), hidden_size,
                                       device=device, dtype=dtype))

            # Compute sequences separately
            seq_outs = []
            seq_hiddens = []
            for i, l in enumerate(lengths):
                hidden_i = maybe_index_tuple(hidden0, i)
                out, hid = lstm2(x[:l, i:i + 1], hidden_i)
                out_pad = pad(out, max_length)
                seq_outs.append(out_pad)
                seq_hiddens.append(hid)
            seq_out = torch.cat(seq_outs, 1)
            seq_hidden = tuple(torch.cat(hids, 1) for hids in zip(*seq_hiddens))

            # Use packed format
            packed = rnn_utils.pack_padded_sequence(x, lengths, enforce_sorted=enforce_sorted)
            packed_out, packed_hidden = lstm(packed, hidden0)
            unpacked, unpacked_len = rnn_utils.pad_packed_sequence(packed_out)

            # Check forward
            prec = dtype2prec_DONTUSE[dtype]
            self.assertEqual(packed_hidden, seq_hidden, atol=prec, rtol=0)
            self.assertEqual(unpacked, seq_out, atol=prec, rtol=0)
            self.assertEqual(unpacked_len, lengths, atol=prec, rtol=0)

            # Check backward
            seq_out.sum().backward()
            grad_x = x_leaf.grad.data.clone()
            x_leaf.grad.data.zero_()
            unpacked.sum().backward()

            self.assertEqual(x_leaf.grad, grad_x, atol=dtype2prec_DONTUSE[dtype], rtol=0)
            for p1, p2 in zip(lstm.parameters(), lstm2.parameters()):
                prec = dtype2prec_DONTUSE[dtype]
                if dtype == torch.float16:
                    prec = 4e-2
                self.assertEqual(p1.grad, p2.grad, atol=prec, rtol=0)

        tests = [
            # enforce_sorted, lengths
            [True, [5]],
            [False, [5]],
            [True, [10, 10, 6, 2, 2, 1, 1]],
            [False, [10, 10, 6, 2, 2, 1, 1]],
            [False, [2, 1, 3, 2, 10, 5, 3]],
        ]

        for enforce_sorted, seq_lens, in tests:
            for use_default_hiddens in (True, False):
                for proj_size in [0, 2]:
                    check_lengths(seq_lens, enforce_sorted, use_default_hiddens, proj_size)

    def _test_batchnorm_update_stats(self, device, dtype=torch.float):
        module = nn.BatchNorm1d(3).to(device, dtype)

        data = torch.rand(4, 3, device=device, dtype=dtype)

        # training pass
        old_running_mean = module.running_mean.clone()
        old_running_var = module.running_var.clone()
        old_num_batches_tracked = module.num_batches_tracked.clone()
        module(data)
        self.assertNotEqual(old_running_mean, module.running_mean)
        self.assertNotEqual(old_running_var, module.running_var)
        self.assertEqual(old_num_batches_tracked + 1, module.num_batches_tracked)

        # eval pass
        module.eval()
        old_running_mean = module.running_mean.clone()
        old_running_var = module.running_var.clone()
        old_num_batches_tracked = module.num_batches_tracked.clone()
        module(data)
        self.assertEqual(old_running_mean, module.running_mean)
        self.assertEqual(old_running_var, module.running_var)
        self.assertEqual(old_num_batches_tracked, module.num_batches_tracked)

    def test_batchnorm_update_stats(self, device):
        self._test_batchnorm_update_stats(device)

        if self.device_type == 'cuda' and self.has_cudnn():
            with torch.backends.cudnn.flags(enabled=False):
                self._test_batchnorm_update_stats(device)

    def test_multi_margin_loss_errors(self, device):
        self.assertRaises(RuntimeError,
                          lambda: nn.functional.multi_margin_loss(torch.randn(5, device=device),
                                                                  torch.zeros(3, device=device)))

    def _test_bfloat16_ops(self, op, device, inp_dims=(), prec=1e-2):
        # fp32 compute
        input1 = torch.randn(inp_dims, dtype=torch.float32, device=device, requires_grad=True)
        out1 = op(input1)
        grad_input1 = torch.randn_like(out1, device=device)
        out1.backward(grad_input1)

        # bfloat16 compute
        op_bfp16 = op.bfloat16()
        input2 = input1.detach().bfloat16().requires_grad_()
        grad_input2 = grad_input1.bfloat16()
        out2 = op_bfp16(input2)
        out2.backward(grad_input2)

        self.assertEqual(out1, out2, atol=prec, rtol=0, exact_dtype=False)
        self.assertEqual(input1.grad.data, input2.grad.data, atol=prec, rtol=0, exact_dtype=False)

    @onlyCUDA
    def test_activations_bfloat16(self, device):
        self._test_bfloat16_ops(torch.nn.ReLU(), device, inp_dims=(5), prec=1e-2)
        self._test_bfloat16_ops(torch.nn.Threshold(0.1, 20), device, inp_dims=(5), prec=1e-2)
        self._test_bfloat16_ops(torch.nn.ELU(), device, inp_dims=(5), prec=1e-2)
        self._test_bfloat16_ops(torch.nn.Softplus(), device, inp_dims=(5), prec=1e-2)
        self._test_bfloat16_ops(torch.nn.Hardshrink(), device, inp_dims=(5), prec=1e-2)
        self._test_bfloat16_ops(torch.nn.Softshrink(), device, inp_dims=(5), prec=1e-2)
        self._test_bfloat16_ops(torch.nn.LeakyReLU(), device, inp_dims=(5), prec=1e-2)

    @onlyCUDA
    def test_pooling_bfloat16(self, device):
        self._test_bfloat16_ops(torch.nn.AvgPool1d(3, stride=2), device, inp_dims=(8, 4, 16), prec=0.05)
        self._test_bfloat16_ops(torch.nn.AvgPool2d(3, stride=2), device, inp_dims=(8, 4, 16, 16), prec=0.05)
        self._test_bfloat16_ops(torch.nn.AvgPool3d(3, stride=2), device, inp_dims=(8, 4, 16, 16, 16), prec=0.05)
        self._test_bfloat16_ops(torch.nn.AdaptiveAvgPool1d(3), device, inp_dims=(8, 4, 16), prec=0.05)
        self._test_bfloat16_ops(torch.nn.AdaptiveAvgPool2d((3, 5)), device, inp_dims=(8, 4, 16, 16), prec=0.05)
        self._test_bfloat16_ops(torch.nn.AdaptiveAvgPool3d((3, 5, 7)), device, inp_dims=(8, 4, 16, 16, 16), prec=0.05)

    @onlyCUDA
    def test_softmax_bfloat16(self, device):
        self._test_bfloat16_ops(torch.nn.Softmax(dim=1), device, inp_dims=(16, 32), prec=1e-2)

    @onlyCUDA
    @skipCUDAIfRocm
    @skipCUDAIfCudnnVersionLessThan(7603)
    @dtypes(torch.half, torch.float)
    def test_conv_cudnn_nhwc(self, device, dtype):
        def helper(n, c, h, w, out_channels, kernel_size, groups):
            input = torch.randint(-3, 3, (n, c, h, w), dtype=dtype, device=device)\
                .to(memory_format=torch.channels_last)
            input.requires_grad_()
            conv = nn.Conv2d(c, out_channels, kernel_size, groups=groups)\
                .to(device='cuda', dtype=dtype, memory_format=torch.channels_last)
            for p in conv.parameters():
                p.data = torch.randint_like(p, -3, 3)

            # use FP64 channels-first conv as reference
            ref_input = input.detach().clone().contiguous().double().requires_grad_()
            ref_conv = nn.Conv2d(c, out_channels, kernel_size, groups=groups)
            # load_state_dict will restore the stride & memory_layout on ref_conv.weight.
            ref_conv.load_state_dict(conv.state_dict())
            ref_conv = ref_conv.to(device='cuda', dtype=torch.double, memory_format=torch.contiguous_format)

            out = conv(input)
            ref_out = ref_conv(ref_input)

            grad = torch.randint_like(out, -3, 3)
            ref_grad = grad.detach().clone().double().contiguous()

            out.backward(grad)
            ref_out.backward(ref_grad)

            self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
            self.assertTrue(input.grad.is_contiguous(memory_format=torch.channels_last))
            self.assertTrue(conv.weight.grad.is_contiguous(memory_format=torch.channels_last))

            self.assertTrue(ref_out.is_contiguous())
            self.assertTrue(ref_input.grad.is_contiguous())
            self.assertTrue(ref_conv.weight.grad.is_contiguous())

            self.assertEqual(out, ref_out, exact_dtype=False)
            self.assertEqual(conv.weight.grad, ref_conv.weight.grad, exact_dtype=False)
            self.assertEqual(conv.bias.grad, ref_conv.bias.grad, exact_dtype=False)
            self.assertEqual(input.grad, ref_input.grad, exact_dtype=False)

        helper(2, 8, 4, 4, out_channels=4, kernel_size=3, groups=1)
        helper(2, 8, 4, 4, out_channels=8, kernel_size=3, groups=8)
        helper(1, 16, 56, 56, out_channels=16, kernel_size=3, groups=1)
        helper(1, 16, 56, 56, out_channels=16, kernel_size=3, groups=16)

    @onlyCUDA
    @skipCUDAIfRocm
    @skipCUDAIfCudnnVersionLessThan(8005)
    @dtypes(torch.half, torch.float)
    def test_conv_cudnn_ndhwc(self, device, dtype):
        def helper(n, c, d, h, w, out_channels, kernel_size, groups):
            input = torch.randint(-2, 2, (n, c, d, h, w), dtype=dtype, device=device)\
                .to(memory_format=torch.channels_last_3d)
            input.requires_grad_()
            conv = nn.Conv3d(c, out_channels, kernel_size, groups=groups)\
                .to(device='cuda', dtype=dtype, memory_format=torch.channels_last_3d)
            for p in conv.parameters():
                p.data = torch.randint_like(p, -2, 2)

            # use FP64 channels-first conv as reference
            ref_input = input.detach().clone().contiguous().double().requires_grad_()
            ref_conv = nn.Conv3d(c, out_channels, kernel_size, groups=groups)
            # load_state_dict will restore the stride & memory_layout on ref_conv.weight.
            ref_conv.load_state_dict(conv.state_dict())
            ref_conv = ref_conv.to(device='cuda', dtype=torch.double, memory_format=torch.contiguous_format)

            out = conv(input)
            ref_out = ref_conv(ref_input)

            grad = torch.randint_like(out, -2, 2)
            ref_grad = grad.detach().clone().double().contiguous()

            out.backward(grad)
            ref_out.backward(ref_grad)

            self.assertTrue(out.is_contiguous(memory_format=torch.channels_last_3d))
            self.assertTrue(input.grad.is_contiguous(memory_format=torch.channels_last_3d))
            self.assertTrue(conv.weight.grad.is_contiguous(memory_format=torch.channels_last_3d))

            self.assertTrue(ref_out.is_contiguous())
            self.assertTrue(ref_input.grad.is_contiguous())
            self.assertTrue(ref_conv.weight.grad.is_contiguous())

            self.assertEqual(out, ref_out, exact_dtype=False)
            self.assertEqual(conv.weight.grad, ref_conv.weight.grad, exact_dtype=False)
            self.assertEqual(conv.bias.grad, ref_conv.bias.grad, exact_dtype=False)
            self.assertEqual(input.grad, ref_input.grad, exact_dtype=False)

        helper(2, 8, 4, 4, 4, out_channels=4, kernel_size=3, groups=1)
        helper(2, 8, 4, 4, 4, out_channels=8, kernel_size=3, groups=8)
        helper(1, 16, 18, 18, 18, out_channels=16, kernel_size=3, groups=1)
        helper(1, 16, 18, 18, 18, out_channels=16, kernel_size=3, groups=16)

    def _run_conv(self, layer, device, inp, grad, ref_conv, ref_input, ref_out,
                  input_format, weight_format, grad_format, output_format):
        conv = layer(inp.size(1), grad.size(1),
                     ref_conv.weight.size(2)).float().to(device)
        # load_state_dict will restore the stride & memory_layout on ref_conv.weight.
        conv.load_state_dict(ref_conv.state_dict())
        weight_data = conv.weight.detach().clone().contiguous(memory_format=weight_format)
        conv.weight.data = weight_data.resize_(weight_data.size(), memory_format=weight_format)
        input = inp.clone().contiguous(memory_format=input_format)
        input.resize_(input.size(), memory_format=input_format)
        input = input.requires_grad_()
        grad = grad.contiguous(memory_format=grad_format)
        grad.resize_(grad.size(), memory_format=grad_format)
        out = conv(input)
        out.backward(grad)
        self.assertTrue(out.is_contiguous(memory_format=output_format))
        self.assertEqual(out, ref_out)
        self.assertEqual(conv.weight.grad, ref_conv.weight.grad)
        self.assertEqual(conv.bias.grad, ref_conv.bias.grad)
        self.assertEqual(input.grad, ref_input.grad)

    def _test_conv_cudnn_nhwc_nchw(self, layer, n, c, h, w, k, filter_size, device):
        data = torch.randint(1, 10, (n, c, h, w), dtype=torch.float32, device=device)
        ref_input = data.clone().contiguous().requires_grad_(True)
        ref_conv = layer(c, k, filter_size).float().to(device)
        ref_out = ref_conv(ref_input)
        grad = torch.randint(1, 10, ref_out.size(), dtype=torch.float32, device="cuda")
        ref_out.backward(grad)

        for w_f in [torch.contiguous_format, torch.channels_last]:
            for g_f in [torch.contiguous_format, torch.channels_last]:
                for input_format in [torch.contiguous_format, torch.channels_last]:
                    output_format = torch.contiguous_format
                    # Older versions of CudNN have Channels Last support disabled
                    if torch.backends.cudnn.version() >= 7603:
                        if input_format == torch.channels_last:
                            output_format = torch.channels_last
                        # This is because we have N111 weight that cannot handle
                        # the ambiguous memory_format
                        if w_f == torch.channels_last:
                            if layer == nn.Conv2d and filter_size * c != 1:
                                output_format = torch.channels_last
                            if layer == nn.ConvTranspose2d and filter_size * k != 1:
                                output_format = torch.channels_last
                    self._run_conv(layer, device, data, grad, ref_conv, ref_input,
                                   ref_out, input_format, w_f, g_f, output_format)

    @onlyCUDA
    @skipCUDAIfRocm
    @skipCUDAIfCudnnVersionLessThan(7603)
    @tf32_on_and_off(0.05)
    def test_conv_cudnn_mismatch_memory_format(self, device):
        configs = [
            [4, 2, 8, 8, 4, 2],
            [4, 1, 8, 8, 4, 2],
            [1, 1, 8, 8, 4, 2],
            [4, 2, 2, 8, 4, 1],
            [4, 2, 1, 8, 4, 1],
            [4, 2, 8, 8, 4, 1],
            [4, 1, 8, 8, 4, 1],
        ]
        for n, c, h, w, k, filter_size in configs:
            self._test_conv_cudnn_nhwc_nchw(nn.Conv2d, n, c, h, w, k, filter_size, device)
            self._test_conv_cudnn_nhwc_nchw(nn.ConvTranspose2d, n, c, h, w, k, filter_size, device)

    # torch.half is erroring out on Windows with CUDA 10.1 + cuDNN 7.6.4
    # returning CUDNN_STATUS_BAD_PARAM
    # Disabling that specific test for now [see issue # 33918]
    @onlyCUDA
    @skipCUDAIfNoCudnn
    @dtypes(torch.float, torch.double)
    def test_conv_cudnn_nhwc_support(self, device, dtype):
        input = torch.randn((1, 16, 1, 1), dtype=dtype, device="cuda", requires_grad=True)
        weight = torch.randn((8, 16, 3, 3), dtype=dtype, device="cuda", requires_grad=True)
        weight = weight.to(memory_format=torch.channels_last)
        o = torch.conv2d(input, weight, None, (2, 1), (1, 1), (1, 1), 1)
        self.assertTrue(o.is_contiguous(memory_format=torch.channels_last))
        o.sum().backward()

    # Test that faster algorithms used for inference produce the same results
    # Validates depthwise3x3 bug reported in https://github.com/pytorch/pytorch/issues/60176
    @onlyCPU
    @dtypes(torch.float)
    def test_conv2d_no_grad(self, device, dtype):
        for batch in [1, 2, 3]:
            for groups in [1, 2, 4]:
                input = torch.rand(batch, groups, 8, 8, dtype=dtype, device=device)
                m = nn.Conv2d(groups, 8, kernel_size=(3, 3), groups=groups, dtype=dtype, device=device)
                with torch.no_grad():
                    output_ng = m(input)
                output = m(input)
                self.assertEqual(output, output_ng, rtol=1e-2, atol=1e-5)

    @onlyCUDA
    @skipCUDAIfRocm
    @skipCUDAIfCudnnVersionLessThan(7603)
    def test_convert_conv2d_weight_memory_format(self, device):
        input = torch.randint(1, 10, (2, 8, 4, 4), dtype=torch.float32, device=device)
        model = nn.Sequential(
            nn.Conv2d(8, 4, 3),
            nn.BatchNorm2d(4)).to(device).float()
        for memory_format in [torch.channels_last, torch.contiguous_format]:
            model = nn.utils.convert_conv2d_weight_memory_format(model, memory_format)
            out = model(input)
            self.assertTrue(out.is_contiguous(memory_format=memory_format))

        model = nn.Sequential(
            nn.ConvTranspose2d(8, 4, 3),
            nn.BatchNorm2d(4)).to(device).float()
        for memory_format in [torch.channels_last, torch.contiguous_format]:
            model = nn.utils.convert_conv2d_weight_memory_format(model, memory_format)
            out = model(input)
            self.assertTrue(out.is_contiguous(memory_format=memory_format))

    def test_nll_loss_mismatched_batch(self, device):
        x = torch.randn((10, 3), requires_grad=True, device=device)
        # t should have size (10,)
        t = torch.zeros((3,), dtype=torch.int64, device=device)
        with self.assertRaisesRegex(ValueError, 'Expected.*batch_size'):
            F.nll_loss(x, t)

    def test_nll_loss_out_of_bounds_ignore_index(self, device):
        x = torch.randn(6, 3, requires_grad=True, device=device)
        t = torch.tensor([0, 1, 255, 0, 1, 2], dtype=torch.int64, device=device)
        for reduction in ['mean', 'none']:
            F.nll_loss(x, t, ignore_index=255, reduction=reduction).sum().backward()

    def test_nll_loss_invalid_target_dim(self, device):
        x = torch.randn((10, 3), device=device)
        t = torch.zeros((10, 2), dtype=torch.int64, device=device)
        with self.assertRaisesRegex(RuntimeError, "1D target tensor expected"):
            F.nll_loss(x, t)

    def test_nll_loss_invalid_weights(self, device):
        x = torch.randn((10, 3), device=device)
        t = torch.empty(10, dtype=torch.int64, device=device).random_(0, 3)
        invalid_weights = [
            torch.randn(4, device=device),
            torch.randn(1, 3, device=device),
        ]
        msg = "weight tensor should be defined either for all 3 classes or no classes"
        for weight in invalid_weights:
            with self.assertRaisesRegex(RuntimeError, msg):
                F.nll_loss(x, t, weight=weight)

    def _nll_loss_helper(self, input_size, reduction, expected, device):
        input = torch.rand(input_size, requires_grad=True, device=device)
        num_channels = input_size[1]
        target_size = (input_size[0], ) + tuple(input_size[2:])
        target = torch.randint(num_channels, target_size, device=device)

        output = F.nll_loss(input, target, reduction=reduction)
        # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
        self.assertEqualIgnoreType(output, expected)

        output.sum().backward()
        self.assertEqual(input.grad.size(), input.size())

    def test_nll_loss_empty_tensor_reduction_none(self, device):
        self._nll_loss_helper([0, 3], "none", torch.empty([0], device=device), device)
        self._nll_loss_helper([0, 3, 5, 7], "none", torch.empty([0, 5, 7], device=device), device)
        self._nll_loss_helper([2, 3, 0, 7], "none", torch.empty([2, 0, 7], device=device), device)
        self._nll_loss_helper([2, 3, 5, 0], "none", torch.empty([2, 5, 0], device=device), device)
        self._nll_loss_helper([2, 3, 5, 7, 0], "none", torch.empty([2, 5, 7, 0], device=device), device)

    @unittest.skipIf(TEST_WITH_UBSAN, "division-by-zero error with UBSAN")
    def test_nll_loss_empty_tensor_reduction_mean(self, device):
        nan = torch.tensor(float('nan'), device=device)
        self._nll_loss_helper([0, 3], "mean", nan, device)
        self._nll_loss_helper([0, 3, 5, 7], "mean", nan, device)
        self._nll_loss_helper([2, 3, 0, 7], "mean", nan, device)
        self._nll_loss_helper([2, 3, 5, 0], "mean", nan, device)
        self._nll_loss_helper([2, 3, 5, 7, 0], "mean", nan, device)

    def test_nll_loss_empty_tensor_reduction_sum(self, device):
        zero = torch.tensor(0, device=device)
        self._nll_loss_helper([0, 3], "sum", zero, device)
        self._nll_loss_helper([0, 3, 5, 7], "sum", zero, device)
        self._nll_loss_helper([2, 3, 0, 7], "sum", zero, device)
        self._nll_loss_helper([2, 3, 5, 0], "sum", zero, device)
        self._nll_loss_helper([2, 3, 5, 7, 0], "sum", zero, device)

    def test_nll_loss_total_weight_is_zero(self, device):

        def helper(input_size):
            input = torch.ones(input_size, requires_grad=True, device=device)
            num_channels = input_size[1]
            target_size = (input_size[0], ) + tuple(input_size[2:])
            target = torch.zeros(target_size, dtype=torch.long, device=device)
            weight = torch.zeros([num_channels], device=device)
            self.assertEqual(F.nll_loss(input, target, weight).item(), 0)

        helper([2, 3])
        helper([2, 3, 5, 7])
        helper([2, 3, 5, 7, 9])

    def test_nll_loss_byte_target_matches_long(self, device):
        N, C = 10, 4
        input = torch.randn(N, C, device=device, requires_grad=True)
        target = torch.empty(N, dtype=torch.long, device=device).random_(0, C)

        def compute_result_and_gradient(reduction, target_dtype):
            input_ = input.detach()
            input_.requires_grad_()

            prob = F.log_softmax(input_, dim=-1)
            loss = nn.NLLLoss(reduction=reduction)
            result = loss(prob, target.to(target_dtype))
            result.sum().backward()

            return result, input_.grad

        for reduction in ["none", "mean", "sum"]:
            result_long, grad_long = compute_result_and_gradient(reduction, torch.long)
            result_byte, grad_byte = compute_result_and_gradient(reduction, torch.uint8)
            self.assertEqual(result_long, result_byte)
            self.assertEqual(grad_long, grad_byte)

    def test_softshrink_negative(self, device):
        input = torch.randn(5, device=device, requires_grad=True)
        m = torch.nn.Softshrink(-1)
        with self.assertRaisesRegex(RuntimeError,
                                    r'lambda must be greater or equal to 0, but found to be -1\.'):
            m(input)

    def test_unfold(self, device):
        def func(x):
            return F.unfold(x, kernel_size=(3, 3))
        seeds = (13, 256, 811, 43, 7)
        for sd in seeds:
            torch.manual_seed(sd)
            x = torch.randn(1, 1, 5, 5, device=device, requires_grad=True)
            gradcheck(func, [x])
            gradgradcheck(func, [x])

    def test_fold(self, device):
        def func(x):
            return F.fold(x, output_size=(4, 5), kernel_size=(2, 2))
        seeds = (44, 83, 71, 25, 999)
        for sd in seeds:
            torch.manual_seed(sd)
            x = torch.randn(1, 12, 12, device=device, requires_grad=True)
            gradcheck(func, [x])
            gradgradcheck(func, [x])

    def test_logsigmoid_out(self, device):
        # this isn't actually documented, but was broken previously:
        # https://github.com/pytorch/pytorch/issues/36499
        x = torch.randn(2, 3, device=device).t()
        empty_out = torch.randn(0, device=device)
        self.assertEqual(F.logsigmoid(x), F.logsigmoid(x, out=empty_out))

        noncontig_out = torch.randn(2, 3, device=device).t()
        self.assertEqual(F.logsigmoid(x), F.logsigmoid(x, out=noncontig_out))

    def test_maxpool3d_non_square_backward(self, device):
        # previous CUDA routine of this backward calculates kernel launch grid size
        # with last two dimensions interchanged, so the tailing along the longer dim
        # get ignored. Here we test whether every position gets gradient.
        for dim in (2, 3, 4):
            shape = tuple(32 if i != dim else 256 for i in range(4))
            x = torch.randn(shape, device=device, requires_grad=True)
            F.max_pool3d(x, kernel_size=(1, 1, 1)).sum().backward()
            self.assertTrue(torch.allclose(x.grad, torch.ones_like(x.grad)))

    # Check that clip_grad_norm_ raises an error if the total norm of the
    # parameters' gradients is non-finite
    def test_clip_grad_norm_error_if_nonfinite(self, device):
        norms_pos = [0.1, 1, 2, 3.5, inf]
        norms_neg = [-0.1, -1, -2, -3.5]
        norms_except_0 = norms_pos + norms_neg
        norms_all = norms_except_0 + [0]

        # Each entry in test_cases has the following values, in this order:
        #
        # grad_only_one_elem    If True, only one element of the parameter's
        #                       gradient is set to the scalar grad, and the
        #                       rest of the elements are 0. If False, all grad
        #                       elements are equal to the scalar.
        #
        # prefix_finite_grad_param  If True, prefix a parameter that has a grad
        #                           of 1.
        #
        # scalars           Scalars to use as the parameter's grad, through
        #                   multiplication
        #
        # norms_nonfinite   Norm types that should produce nonfinite total norm
        #
        # norms_finite      Norm types that should produce finite total norm
        test_cases = [
            # Test errors from an infinite grad
            (False, False, [inf, -inf], norms_except_0, [0]),
            (False, True, [inf, -inf], norms_pos, norms_neg + [0]),
            (True, False, [inf, -inf], norms_pos, norms_neg + [0]),
            (True, True, [inf, -inf], norms_pos, norms_neg + [0]),

            # Test errors from a NaN grad
            (False, False, [nan], norms_except_0, [0]),
            (False, True, [nan], norms_except_0, [0]),
            (True, False, [nan], norms_except_0, [0]),
            (True, True, [nan], norms_except_0, [0]),

            # Test a grad that should never error
            (False, False, [2e22, -2e22], [], norms_all),
            (False, True, [2e22, -2e22], [], norms_all),
            (True, False, [2e22, -2e22], [], norms_all),
            (True, True, [2e22, -2e22], [], norms_all),

            # Test a grad that will overflow to inf for only some norm orders
            (False, False, [2e200, -2e200], [3.5, 2, -2, -3.5], [inf, 1, 0.1, 0, -1, -0.1]),
            (False, True, [2e200, -2e200], [3.5, 2], norms_neg + [inf, 1, 0.1, 0]),
            (True, False, [2e200, -2e200], [3.5, 2], norms_neg + [inf, 1, 0.1, 0]),
            (True, True, [2e200, -2e200], [3.5, 2], norms_neg + [inf, 1, 0.1, 0]),
        ]

        def gen_parameters(scalar, grad_only_one_elem, prefix_finite_grad_param):
            param = torch.ones(10, dtype=torch.float64, device=device, requires_grad=True)

            if grad_only_one_elem:
                param[1].mul(scalar).sum().backward()
            else:
                param.mul(scalar).sum().backward()

            if prefix_finite_grad_param:
                prefix_param = torch.ones(1, dtype=torch.float64, device=device, requires_grad=True)
                prefix_param.mul(1).sum().backward()
                parameters = [prefix_param, param]
            else:
                parameters = [param]

            return parameters

        def run_test_case(norm_type, error_if_nonfinite, scalar, grad_only_one_elem, prefix_finite_grad_param, is_norm_nonfinite):
            msg = (
                f'norm_type: {norm_type}, ',
                f'error_if_nonfinite: {error_if_nonfinite}, '
                f'scalar: {scalar}, '
                f'grad_only_one_elem: {grad_only_one_elem}, '
                f'prefix_finite_grad_param: {prefix_finite_grad_param}, '
                f'is_norm_nonfinite: {is_norm_nonfinite}')

            parameters = gen_parameters(scalar, grad_only_one_elem, prefix_finite_grad_param)

            # Should only throw an error if the total norm is expected to be
            # nonfinite and `error_if_nonfinite=True`
            if is_norm_nonfinite and error_if_nonfinite:
                error_msg = f'The total norm of order {float(norm_type)} for gradients'

                grads_before = [p.grad.clone() for p in parameters]

                with self.assertRaisesRegex(RuntimeError, error_msg, msg=msg):
                    clip_grad_norm_(parameters, 1, norm_type=norm_type, error_if_nonfinite=True)

                # Grad should not change if error is thrown
                grads_after = [p.grad for p in parameters]
                self.assertEqual(grads_before, grads_after, msg=msg)
            else:
                clip_grad_norm_(parameters, 1, norm_type=norm_type, error_if_nonfinite=error_if_nonfinite)

        for grad_only_one_elem, prefix_finite_grad_param, scalars, norms_nonfinite, norms_finite in test_cases:
            for error_if_nonfinite in [False, True]:
                for norm_type, scalar in product(norms_nonfinite, scalars):
                    run_test_case(norm_type, error_if_nonfinite, scalar, grad_only_one_elem, prefix_finite_grad_param, True)

                for norm_type, scalar in product(norms_finite, scalars):
                    run_test_case(norm_type, error_if_nonfinite, scalar, grad_only_one_elem, prefix_finite_grad_param, False)

    @onlyCUDA
    @deviceCountAtLeast(2)
    def test_clip_grad_norm_multi_device(self, devices):
        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()
                self.layer1 = nn.Linear(10, 10)
                self.layer2 = nn.Linear(10, 10)

        test_model = TestModel()
        test_model.layer1.to(devices[0])
        test_model.layer2.to(devices[1])
        ref_model = TestModel().to(devices[0])
        for norm_type in [2., math.inf]:
            for p in test_model.parameters():
                p.grad = torch.ones_like(p)
            for p in ref_model.parameters():
                p.grad = torch.ones_like(p)
            norm = clip_grad_norm_(test_model.parameters(), 0.5, norm_type=norm_type)
            expected = clip_grad_norm_(ref_model.parameters(), 0.5, norm_type=norm_type)
            self.assertEqual(norm, expected)
            for p, pe in zip(test_model.parameters(), ref_model.parameters()):
                self.assertEqual(p.grad.to(devices[0]), pe.grad)

    def test_elu_inplace_overlap(self, device):
        x = torch.randn((1, 6), device=device).expand((6, 6))
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            F.elu(x, inplace=True)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            F.elu_(x)

    @expectedFailureMeta  # https://github.com/pytorch/pytorch/issues/54897
    def test_hardswish_inplace_overlap(self, device):
        x = torch.randn((1, 6), device=device).expand((6, 6))
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            F.hardswish(x, inplace=True)

    def test_silu_inplace_overlap(self, device):
        x = torch.randn((1, 6), device=device).expand((6, 6))
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            F.silu(x, inplace=True)

    @onlyOnCPUAndCUDA
    def test_mish_inplace_overlap(self, device):
        x = torch.randn((1, 6), device=device).expand((6, 6))
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            F.mish(x, inplace=True)

    def test_softplus_inplace_overlap(self, device):
        x = torch.randn((1, 6), device=device).expand((6, 6))
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            F.softplus(x, out=x)

    def test_softplus_low_threshold(self, device):
        # Ensure gradients are computed correctly with a low threshold.
        model = torch.nn.Softplus(threshold=1).double()
        input = torch.tensor(0.9, device=device, dtype=torch.double,
                             requires_grad=True)
        output = model(input)
        torch.autograd.gradcheck(model, input)

    def test_softshrink_inplace_overlap(self, device):
        x = torch.randn((1, 6), device=device).expand((6, 6))
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            F.softshrink(x, out=x)

    def test_leaky_relu_inplace_overlap(self, device):
        x = torch.randn((1, 6), device=device).expand((6, 6))
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            F.leaky_relu(x, inplace=True)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            F.leaky_relu_(x)

    def test_threshold_inplace_overlap(self, device):
        # Inplace threshold is okay, because it is idempotent
        x = torch.randn((1, 6), device=device).expand((6, 6))
        F.threshold(x, 0.5, 0.5, inplace=True)
        F.threshold_(x, 0.5, 0.5)

    @onlyOnCPUAndCUDA
    def test_triplet_margin_with_distance_loss_default_parity(self, device):
        # Test for `nn.TripletMarginWithDistanceLoss` and
        # `F.triplet_margin_with_distance_loss`.  Checks
        # for parity against the respective non-distance-agnostic
        # implementations of triplet margin loss (``nn.TripletMarginLoss`
        # and `F.triplet_margin_loss`) under *default args*.

        for extra_args in \
                itertools.product((0.5, 1, 1.5), (True, False), ('none', 'mean', 'sum')):
            kwargs = {'margin': extra_args[0], 'swap': extra_args[1], 'reduction': extra_args[2]}

            anchor = torch.randn(5, 10, device=device, requires_grad=True)
            positive = torch.randn(5, 10, device=device, requires_grad=True)
            negative = torch.randn(5, 10, device=device, requires_grad=True)

            # Test forward, functional
            expected = F.triplet_margin_loss(anchor, positive, negative, **kwargs)
            actual = F.triplet_margin_with_distance_loss(anchor, positive, negative, **kwargs)
            self.assertEqual(actual, expected, rtol=1e-6, atol=1e-6)

            # Test forward, module
            loss_ref = nn.TripletMarginLoss(**kwargs)
            loss_op = nn.TripletMarginWithDistanceLoss(**kwargs)
            self.assertEqual(loss_op(anchor, positive, negative),
                             loss_ref(anchor, positive, negative),
                             rtol=1e-6, atol=1e-6)

            # Test backward
            self.assertTrue(gradcheck(lambda a, p, n: F.triplet_margin_with_distance_loss(
                a, p, n, **kwargs), (anchor, positive, negative)))
            self.assertTrue(gradcheck(lambda a, p, n: loss_op(a, p, n),
                            (anchor, positive, negative)))

    @onlyOnCPUAndCUDA
    def test_triplet_margin_with_distance_loss(self, device):
        # Test for parity between `nn.TripletMarginWithDistanceLoss` and
        # `F.triplet_margin_with_distance_loss`.

        pairwise_distance = nn.PairwiseDistance()

        def cosine_distance(x, y):
            return 1.0 - F.cosine_similarity(x, y)

        distance_functions = (pairwise_distance, cosine_distance,
                              lambda x, y: 1.0 - F.cosine_similarity(x, y))

        reductions = ('mean', 'none', 'sum')
        margins = (1.0, 1.5, 0.5)
        swaps = (True, False)

        for distance_fn, reduction, margin, swap \
                in itertools.product(distance_functions, reductions, margins, swaps):
            anchor = torch.randn(5, 10, device=device, requires_grad=True)
            positive = torch.randn(5, 10, device=device, requires_grad=True)
            negative = torch.randn(5, 10, device=device, requires_grad=True)

            # Test backward
            self.assertTrue(gradcheck(lambda a, p, n: F.triplet_margin_with_distance_loss(
                a, p, n, distance_function=distance_fn, reduction=reduction, margin=margin, swap=swap),
                (anchor, positive, negative)))
            loss_op = nn.TripletMarginWithDistanceLoss(distance_function=distance_fn,
                                                       reduction=reduction, margin=margin, swap=swap)
            self.assertTrue(gradcheck(lambda a, p, n: loss_op(
                a, p, n), (anchor, positive, negative)))
            traced_loss_op = torch.jit.trace(loss_op, (anchor, positive, negative))
            self.assertTrue(gradcheck(lambda a, p, n: traced_loss_op(
                a, p, n), (anchor, positive, negative)))

            # Test forward parity
            functional = F.triplet_margin_with_distance_loss(anchor, positive, negative,
                                                             distance_function=distance_fn,
                                                             reduction=reduction, margin=margin, swap=swap)
            modular = loss_op(anchor, positive, negative)
            traced = traced_loss_op(anchor, positive, negative)
            self.assertEqual(functional, modular, atol=1e-6, rtol=1e-6)
            self.assertEqual(traced, modular, atol=1e-6, rtol=1e-6)

    def test_to_complex(self, device):
        m = nn.Linear(3, 5).to(device)
        self.assertIs(m, m.to(device))
        m.to(torch.cfloat)
        self.assertIs(m.weight.dtype, torch.cfloat)
        m.to(torch.cdouble)
        self.assertIs(m.weight.dtype, torch.cdouble)
        m.to(torch.float)
        self.assertIs(m.weight.dtype, torch.float)
        with warnings.catch_warnings(record=True) as w:
            # Trigger warning
            m.to(torch.cfloat)
            # Check warning occurs
            self.assertEqual(len(w), 1)
            self.assertTrue("Complex modules are a new feature" in str(w[-1].message))

    @skipMeta
    @dtypes(torch.float32, torch.float64)
    def test_module_to_empty(self, device, dtype):
        class MyModule(nn.Module):
            def __init__(self, in_features, out_features, device=None, dtype=None):
                super().__init__()
                factory_kwargs = {"device": device, "dtype": dtype}
                self.weight = nn.Parameter(torch.randn(in_features, out_features, **factory_kwargs))

            def forward(self, x):
                return x @ self.weight

        # Test meta module instantiation.
        input = torch.randn(5, 10, device=device, dtype=dtype)
        m = MyModule(10, 1, device='meta', dtype=dtype)
        m(input)

        # Test materializing meta module on a real device.
        m.to_empty(device=device)
        m(input)
        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(m.weight)
        m(input)

        # Test creating meta module from materialized module.
        m.to_empty(device='meta')
        m(input)

    @skipMeta
    def test_skip_init(self, device):
        torch.manual_seed(1)
        m_initialized = torch.nn.Linear(5, 1)
        m_initialized.to(device)

        torch.manual_seed(1)
        m_uninitialized = torch.nn.utils.skip_init(torch.nn.Linear, 5, 1, device=device)

        self.assertEqual(m_initialized.weight.device, m_uninitialized.weight.device)
        self.assertFalse(torch.allclose(m_initialized.weight, m_uninitialized.weight))

class TestModuleGlobalHooks(TestCase):

    def tearDown(self):
        nn.modules.module._global_backward_hooks = OrderedDict()
        nn.modules.module._global_forward_hooks = OrderedDict()
        nn.modules.module._global_forward_pre_hooks = OrderedDict()

    def test_module_global_hooks(self):
        module = nn.Sigmoid

        module_1 = module()
        module_2 = module()
        module_3 = module()

        input = torch.ones(5, 5, requires_grad=True)

        counter = {
            'forwards': 0,
            'backwards': 0
        }

        def fw_hook(inc, h_module, input, output):
            self.assertIsInstance(input, tuple)
            self.assertTrue(isinstance(output, torch.Tensor))
            self.assertTrue(isinstance(h_module, module))
            self.assertEqual(input[0], torch.ones(5, 5))
            self.assertEqual(output, torch.empty(5, 5).fill_(1 / (1 + 1 / math.e)))
            counter['forwards'] += inc

        def bw_hook(inc, h_module, grad_input, grad_output):
            self.assertIsInstance(grad_input, tuple)
            self.assertIsInstance(grad_output, tuple)
            self.assertTrue(isinstance(h_module, module))
            self.assertEqual(grad_output[0], torch.ones(5, 5) * 2)
            counter['backwards'] += inc

        test_fwd = nn.modules.module.register_module_forward_hook(lambda *args: fw_hook(1, *args))

        module_1(input)
        module_2(input)
        module_3(input)
        self.assertEqual(counter['forwards'], 3)
        self.assertEqual(counter['backwards'], 0)

        test_bwd = nn.modules.module.register_module_backward_hook(
            lambda *args: bw_hook(1, *args))

        output_1 = module_1(input)
        output_2 = module_2(input)
        output_3 = module_3(input)
        self.assertEqual(counter['forwards'], 6)
        self.assertEqual(counter['backwards'], 0)

        output_1.backward(torch.ones(5, 5) * 2, retain_graph=True)
        output_2.backward(torch.ones(5, 5) * 2, retain_graph=False)
        output_3.backward(torch.ones(5, 5) * 2, retain_graph=False)
        self.assertEqual(counter['forwards'], 6)
        self.assertEqual(counter['backwards'], 3)

        output_1.backward(torch.ones(5, 5) * 2, retain_graph=True)
        self.assertEqual(counter['forwards'], 6)
        self.assertEqual(counter['backwards'], 4)

        test2_fwd = nn.modules.module.register_module_forward_hook(lambda *args: fw_hook(2, *args))

        output = module_1(input)
        output = module_2(input)
        output = module_3(input)
        self.assertEqual(counter['forwards'], 15)
        self.assertEqual(counter['backwards'], 4)

        test2_bwd = nn.modules.module.register_module_backward_hook(lambda *args: bw_hook(2, *args))

        module_1(input).backward(torch.ones(5, 5) * 2)
        self.assertEqual(counter['forwards'], 18)
        self.assertEqual(counter['backwards'], 7)

        test2_bwd.remove()

        module_2(input).backward(torch.ones(5, 5) * 2)
        self.assertEqual(counter['forwards'], 21)
        self.assertEqual(counter['backwards'], 8)

        test2_fwd.remove()

        module_3(input).backward(torch.ones(5, 5) * 2)
        self.assertEqual(counter['forwards'], 22)
        self.assertEqual(counter['backwards'], 9)

        test_fwd.remove()
        test_bwd.remove()

    def test_module_global_hook_invalid_outputs(self):
        module = nn.Sigmoid()
        input = torch.randn(5, 5, requires_grad=True)

        def bw_fail1(self, grad_input, grad_output):
            return grad_input[:-1]

        def bw_fail2(self, grad_input, grad_output):
            return grad_input + (torch.randn(2, 2),)

        with nn.modules.module.register_module_backward_hook(bw_fail1):
            with self.assertRaisesRegex(RuntimeError, 'got 0, but expected 1'):
                module(input).sum().backward()

        with nn.modules.module.register_module_backward_hook(bw_fail2):
            with self.assertRaisesRegex(RuntimeError, 'got 2, but expected 1'):
                module(input).sum().backward()

    def test_module_backward_global_hook_writeable(self):
        module = nn.Sigmoid()
        input = torch.randn(5, 5, requires_grad=True)
        sig_x = torch.sigmoid(input)

        def bw_hook(module, grad_input, grad_output):
            for grad in grad_input:
                self.assertTrue(isinstance(grad, torch.Tensor))
            for grad in grad_output:
                self.assertTrue(isinstance(grad, torch.Tensor))
            return tuple(gi * 2 for gi in grad_input)

        nn.modules.module.register_module_backward_hook(bw_hook)
        module(input).backward(torch.ones(5, 5))
        expected_grad = sig_x * (1 - sig_x) * 2
        self.assertEqual(input.grad, expected_grad)

    def test_module_global_forward_preforward_hook_writeable(self):
        module = nn.Sigmoid()
        input = torch.randn(5, 5, requires_grad=True)
        sig_x = torch.sigmoid(input)

        def forward_pre_hook(m, input):
            return torch.nn.functional.relu(input[0])

        def forward_hook(m, input, output):
            return -output

        nn.modules.module.register_module_forward_pre_hook(forward_pre_hook)
        nn.modules.module.register_module_forward_hook(forward_hook)
        output = module(input)
        expected_res = -torch.sigmoid(torch.nn.functional.relu(input))
        self.assertEqual(output, expected_res)
        output.backward(torch.ones(5, 5) * 2, retain_graph=True)
        mask = (input > 0).double()
        expected_grad = -sig_x * (1 - sig_x) * 2 * mask
        self.assertEqual(input.grad, expected_grad)

    def test_module_forward_preforward_hook_removable(self):
        """
        This test is to test when multiple pre-forward hook functions can be
        registered successfully and used correctly, if the handle can be removable
        during the pre-forward hook function call.
        """
        module = nn.Sigmoid()

        def removable_hook(m, input):
            nonlocal handle
            handle.remove()
            return input

        def removable_hook_2(m, input):
            nonlocal handle_2
            handle_2.remove()
            return input

        handle = module.register_forward_pre_hook(removable_hook)
        handle_2 = module.register_forward_pre_hook(removable_hook_2)

        # make sure hook register is successful
        self.assertEqual(len(handle.hooks_dict_ref()), 2)
        self.assertEqual(len(handle_2.hooks_dict_ref()), 2)

        input = torch.randn(2, 2)
        output = module(input)
        self.assertTrue(torch.allclose(torch.sigmoid(input), output))

        # make sure hook removal is successful
        self.assertFalse(handle.id in handle.hooks_dict_ref())
        self.assertFalse(handle_2.id in handle.hooks_dict_ref())
        self.assertEqual(len(handle.hooks_dict_ref()), 0)
        self.assertEqual(len(handle_2.hooks_dict_ref()), 0)

    def test_module_forward_forward_hook_removable(self):
        """
        This test is to test when multiple forward hook functions can be registered
        successfully and used correctly, if the handle can be removable during the
        forward hook function call.
        """
        module = nn.Sigmoid()

        def removable_hook(m, input, output):
            nonlocal handle
            handle.remove()
            return output

        def removable_hook_2(m, input, output):
            nonlocal handle_2
            handle_2.remove()
            return output

        handle = module.register_forward_hook(removable_hook)
        handle_2 = module.register_forward_hook(removable_hook_2)

        # make sure hook register is successful
        self.assertEqual(len(handle.hooks_dict_ref()), 2)
        self.assertEqual(len(handle_2.hooks_dict_ref()), 2)

        input = torch.randn(2, 2)
        output = module(input)
        self.assertTrue(torch.allclose(torch.sigmoid(input), output))

        # make sure hook removal is successful
        self.assertFalse(handle.id in handle.hooks_dict_ref())
        self.assertFalse(handle_2.id in handle.hooks_dict_ref())
        self.assertEqual(len(handle.hooks_dict_ref()), 0)
        self.assertEqual(len(handle_2.hooks_dict_ref()), 0)

    def test_global_and_local_hooks_order(self):
        module = nn.Sigmoid()

        global_forward_pre_called = False
        local_forward_pre_called = False
        global_forward_called = False
        local_forward_called = False
        global_backward_called = False
        local_backward_called = False

        def global_forward_pre_hook(m, input):
            nonlocal global_forward_pre_called
            self.assertTrue(not local_forward_pre_called)
            global_forward_pre_called = True
            return input

        def local_forward_pre_hook(m, input):
            nonlocal local_forward_pre_called
            self.assertTrue(global_forward_pre_called)
            local_forward_pre_called = True
            return input

        def global_forward_hook(m, input, output):
            nonlocal global_forward_called
            self.assertTrue(not local_forward_called)
            global_forward_called = True
            return output

        def local_forward_hook(m, input, output):
            nonlocal local_forward_called
            self.assertTrue(global_forward_called)
            local_forward_called = True
            return output

        def global_backward_hook(m, input, output):
            nonlocal global_backward_called
            self.assertTrue(not local_backward_called)
            global_backward_called = True
            return input

        def local_backward_hook(m, input, output):
            nonlocal local_backward_called
            self.assertTrue(global_backward_called)
            local_backward_called = True
            return input

        input = torch.randn(5, 5, requires_grad=True)
        nn.modules.module.register_module_forward_pre_hook(global_forward_pre_hook)
        module.register_forward_pre_hook(local_forward_pre_hook)
        nn.modules.module.register_module_forward_hook(global_forward_hook)
        module.register_forward_hook(local_forward_hook)
        nn.modules.module.register_module_backward_hook(global_backward_hook)
        module.register_backward_hook(local_backward_hook)

        output = module(input)
        self.assertTrue(local_forward_called and local_forward_pre_called and global_forward_called and global_forward_pre_called)

        output.backward(torch.ones(5, 5), retain_graph=True)
        self.assertTrue(local_backward_called and global_backward_called)


class LazyModule(torch.nn.modules.lazy.LazyModuleMixin, torch.nn.Module):
    pass


class TestLazyModules(TestCase):

    @suppress_warnings
    def test_lazy_module_parameter(self):
        module = LazyModule()
        module.register_parameter('test_param', UninitializedParameter())
        self.assertTrue(module.has_uninitialized_params())
        state_dict = module.state_dict()
        self.assertIsInstance(state_dict['test_param'], UninitializedParameter)
        new_module = LazyModule()
        # An error is raised when there is an attempt to replace an existing parameter
        # with an uninitialized one
        new_module.register_parameter('test_param', nn.Parameter(torch.ones(5, 5)))
        with self.assertRaisesRegex(RuntimeError, 'shape of an uninitialized'):
            new_module.load_state_dict(state_dict)
        # Uninitialized parameters are overriden when the state dict to be loaded contains a valid one
        new_module = LazyModule()
        new_module.register_parameter('test_param', nn.Parameter(torch.ones(5, 5)))
        module.load_state_dict(new_module.state_dict())
        self.assertEqual(module.test_param, torch.ones((5, 5)))

        # Uninitialized parameters are left unchanged
        module = LazyModule()
        module.register_parameter('test_param', UninitializedParameter())
        self.assertTrue(module.has_uninitialized_params())

        new_module = LazyModule()
        new_module.register_parameter('test_param', UninitializedParameter())
        module.load_state_dict(new_module.state_dict())
        self.assertTrue(module.has_uninitialized_params())

    @suppress_warnings
    def test_lazy_module_buffer(self):
        module = LazyModule()
        module.register_buffer('test_buffer', UninitializedBuffer())
        self.assertTrue(module.has_uninitialized_params())
        state_dict = module.state_dict()
        self.assertIsInstance(state_dict['test_buffer'], UninitializedBuffer)
        new_module = LazyModule()
        # An error is raised when there is an attempt to replace an existing parameter
        # with an uninitialized one
        new_module.register_buffer('test_buffer', torch.ones(5, 5))
        with self.assertRaisesRegex(RuntimeError, 'shape of an uninitialized'):
            new_module.load_state_dict(state_dict)
        # Uninitialized parameters are overriden when the state dict to be loaded contains a valid one
        new_module = LazyModule()
        new_module.register_buffer('test_buffer', torch.ones(5, 5))
        module.load_state_dict(new_module.state_dict())
        self.assertEqual(module.test_buffer, torch.ones((5, 5)))

        # Uninitialized parameters are left unchanged
        module = LazyModule()
        module.register_buffer('test_buffer', UninitializedBuffer())
        self.assertTrue(module.has_uninitialized_params())

        new_module = LazyModule()
        new_module.register_buffer('test_buffer', UninitializedBuffer())
        module.load_state_dict(new_module.state_dict())
        module.load_state_dict(new_module.state_dict())
        self.assertTrue(module.has_uninitialized_params())

    @suppress_warnings
    def test_lazy_module_jit_param(self):
        module = LazyModule()
        module.register_parameter('test_param', UninitializedParameter())
        self.assertTrue(module.has_uninitialized_params())
        with self.assertRaisesRegex(RuntimeError, 'run a forward pass'):
            torch.jit.script(module)

    @suppress_warnings
    def test_lazy_module_jit_buffer(self):
        module = LazyModule()
        module.register_buffer('test_buffer', UninitializedBuffer())
        self.assertTrue(module.has_uninitialized_params())
        with self.assertRaisesRegex(RuntimeError, 'run a forward pass'):
            torch.jit.script(module)

    @suppress_warnings
    def test_lazy_share_memory_param(self):
        module = LazyModule()
        module.register_parameter('test_param', UninitializedParameter())
        self.assertTrue(module.has_uninitialized_params())
        with self.assertRaisesRegex(RuntimeError, 'share memory on an uninitialized'):
            module.share_memory()

    @suppress_warnings
    def test_lazy_share_memory_buffer(self):
        module = LazyModule()
        module.register_buffer('test_buffer', UninitializedBuffer())
        self.assertTrue(module.has_uninitialized_params())
        with self.assertRaisesRegex(RuntimeError, 'share memory on an uninitialized'):
            module.share_memory()

    @suppress_warnings
    def test_linear(self):
        module = nn.LazyLinear(10)
        self.assertIsInstance(module.weight, UninitializedParameter)
        self.assertIsInstance(module.bias, UninitializedParameter)
        input = torch.ones(5, 5)
        module(input)
        self.assertIsInstance(module, nn.Linear)
        self.assertNotIsInstance(module, nn.LazyLinear)
        self.assertTrue(module.weight.shape == (10, 5))
        self.assertTrue(module.bias.shape == (10,))
        y = module(input)
        self.assertTrue(torch.equal(torch.nn.functional.linear(input, module.weight, module.bias), y))

    @suppress_warnings
    def test_lazy_linear_pickle(self):
        module = nn.LazyLinear(10)
        self.assertIsInstance(module.weight, UninitializedParameter)
        self.assertIsInstance(module.bias, UninitializedParameter)
        module = pickle.loads(pickle.dumps(module))
        self.assertIsInstance(module, nn.LazyLinear)
        self.assertIsInstance(module.weight, UninitializedParameter)
        self.assertIsInstance(module.bias, UninitializedParameter)
        input = torch.ones(5, 5)
        module(input)  # fully materialized
        new_module = pickle.loads(pickle.dumps(module))
        self.assertIsInstance(new_module, nn.Linear)
        self.assertNotIsInstance(new_module, nn.LazyLinear)
        self.assertTrue(new_module.weight.shape == (10, 5))
        self.assertNotIsInstance(new_module.weight, UninitializedParameter)
        self.assertTrue(new_module.bias.shape == (10,))
        self.assertNotIsInstance(new_module.bias, UninitializedParameter)

    @suppress_warnings
    def test_linear_state(self):
        module = nn.Linear(5, 10)
        lazy_module = nn.LazyLinear(10)
        lazy_module.load_state_dict(module.state_dict())
        # Parameters have been initialized but the module won't become a full
        # Linear one until the first iteration. This is due to
        # limitations on the state_dict loading logic
        self.assertFalse(lazy_module.has_uninitialized_params())
        self.assertTrue(lazy_module.weight.shape == (10, 5))
        self.assertTrue(lazy_module.bias.shape == (10,))

        module = nn.Linear(5, 10)
        lazy_module = nn.LazyLinear(10)
        with self.assertRaisesRegex(RuntimeError, 'shape of an uninitialized'):
            module.load_state_dict(lazy_module.state_dict())

    def _check_lazy_conv(self, cls, lazy_cls, func, init_args, input_shape,
                         expected_weight_shape, expected_bias_shape):
        module = lazy_cls(*init_args)
        self.assertIsInstance(module.weight, UninitializedParameter)
        if module.bias is not None:
            self.assertIsInstance(module.bias, UninitializedParameter)
        input = torch.ones(*input_shape)
        module(input)
        self.assertIsInstance(module, cls)
        self.assertNotIsInstance(module, lazy_cls)
        self.assertEqual(module.weight.shape, expected_weight_shape)
        if module.bias is not None:
            self.assertEqual(module.bias.shape, expected_bias_shape)
        y = module(input)
        self.assertTrue(torch.equal(func(input, module.weight, module.bias), y))

    def _check_lazy_conv_pickle(self, cls, lazy_cls, init_args, input_shape,
                                expected_weight_shape, expected_bias_shape):
        module = lazy_cls(*init_args)
        self.assertIsInstance(module.weight, UninitializedParameter)
        if module.bias is not None:
            self.assertIsInstance(module.bias, UninitializedParameter)
        module = pickle.loads(pickle.dumps(module))
        self.assertIsInstance(module, lazy_cls)
        self.assertIsInstance(module.weight, UninitializedParameter)
        if module.bias is not None:
            self.assertIsInstance(module.bias, UninitializedParameter)
        input = torch.ones(*input_shape)
        module(input)  # fully materialized
        new_module = pickle.loads(pickle.dumps(module))
        self.assertIsInstance(new_module, cls)
        self.assertNotIsInstance(new_module, lazy_cls)
        self.assertEqual(new_module.weight.shape, expected_weight_shape)
        self.assertNotIsInstance(new_module.weight, UninitializedParameter)
        if new_module.bias is not None:
            self.assertEqual(new_module.bias.shape, expected_bias_shape)
            self.assertNotIsInstance(new_module.bias, UninitializedParameter)

    def _check_lazy_conv_state(self, gen_module, gen_lazy_module,
                               expected_weight_shape, expected_bias_shape):
        module = gen_module()
        lazy_module = gen_lazy_module()
        lazy_module.load_state_dict(module.state_dict())
        # Parameters have been initialized but the module won't become a full
        # Conv one until the first iteration. This is due to
        # limitations on the state_dict loading logic
        self.assertFalse(lazy_module.has_uninitialized_params())
        self.assertEqual(lazy_module.weight.shape, expected_weight_shape)
        if lazy_module.bias is not None:
            self.assertEqual(lazy_module.bias.shape, expected_bias_shape)

        module = gen_module()
        lazy_module = gen_lazy_module()
        with self.assertRaisesRegex(RuntimeError, 'shape of an uninitialized'):
            module.load_state_dict(lazy_module.state_dict())


    def test_lazy_pre_forward_hook(self):
        """
        This test is to test whether lazymodule can register other pre-forward hook
        functions successfully.
        """
        class TestModule(torch.nn.modules.lazy.LazyModuleMixin, torch.nn.Module):
            def __init__(self):
                super().__init__()

            def initialize_parameters(self, input):
                return None

            def forward(self, input):
                return input

        def hook_function(module, input):
            return input[0] + 1

        module = TestModule()
        module.register_forward_pre_hook(hook_function)
        output = module(torch.zeros(2, 2))
        self.assertTrue(torch.allclose(output, torch.ones(2, 2)))

    def test_lazy_forward_hook(self):
        """
        This test is to test whether lazymodule can register other forward hook
        functions successfully.
        """
        class TestModule(torch.nn.modules.lazy.LazyModuleMixin, torch.nn.Module):
            def __init__(self):
                super().__init__()

            def initialize_parameters(self, input):
                return None

            def forward(self, input):
                return input

        def hook_function(module, input, output):
            return input[0] + 1

        module = TestModule()
        module.register_forward_hook(hook_function)
        output = module(torch.zeros(2, 2))
        self.assertTrue(torch.allclose(output, torch.ones(2, 2)))

    @suppress_warnings
    def test_lazy_conv1d(self):
        self._check_lazy_conv(nn.Conv1d, nn.LazyConv1d, torch.nn.functional.conv1d,
                              (32, 2), (192, 16, 50), (32, 16, 2), (32,))

    @suppress_warnings
    def test_lazy_conv1d_pickle(self):
        self._check_lazy_conv_pickle(nn.Conv1d, nn.LazyConv1d, (32, 2), (192, 16, 50),
                                     (32, 16, 2), (32,))

    @suppress_warnings
    def test_lazy_conv1d_state(self):
        self._check_lazy_conv_state(lambda: nn.Conv1d(16, 32, 2),
                                    lambda: nn.LazyConv1d(32, 2),
                                    (32, 16, 2), (32,))

    @suppress_warnings
    def test_lazy_conv2d(self):
        self._check_lazy_conv(nn.Conv2d, nn.LazyConv2d, torch.nn.functional.conv2d,
                              (32, 2), (192, 16, 8, 6), (32, 16, 2, 2), (32,))

    @suppress_warnings
    def test_lazy_conv2d_pickle(self):
        self._check_lazy_conv_pickle(nn.Conv2d, nn.LazyConv2d, (32, 2), (192, 16, 8, 6),
                                     (32, 16, 2, 2), (32,))

    @suppress_warnings
    def test_lazy_conv2d_state(self):
        self._check_lazy_conv_state(lambda: nn.Conv2d(16, 32, 2),
                                    lambda: nn.LazyConv2d(32, 2),
                                    (32, 16, 2, 2), (32,))

    @suppress_warnings
    def test_lazy_conv3d(self):
        self._check_lazy_conv(nn.Conv3d, nn.LazyConv3d, torch.nn.functional.conv3d,
                              (32, 2), (192, 16, 8, 7, 6), (32, 16, 2, 2, 2), (32,))

    @suppress_warnings
    def test_lazy_conv3d_pickle(self):
        self._check_lazy_conv_pickle(nn.Conv3d, nn.LazyConv3d, (32, 2), (192, 16, 8, 7, 6),
                                     (32, 16, 2, 2, 2), (32,))

    @suppress_warnings
    def test_lazy_conv3d_state(self):
        self._check_lazy_conv_state(lambda: nn.Conv3d(16, 32, 2),
                                    lambda: nn.LazyConv3d(32, 2),
                                    (32, 16, 2, 2, 2), (32,))

    @suppress_warnings
    def test_lazy_conv_transposed1d(self):
        self._check_lazy_conv(nn.ConvTranspose1d, nn.LazyConvTranspose1d, torch.nn.functional.conv_transpose1d,
                              (32, 2), (192, 16, 50), (16, 32, 2), (32,))

    @suppress_warnings
    def test_lazy_conv_transpose1d_pickle(self):
        self._check_lazy_conv_pickle(nn.ConvTranspose1d, nn.LazyConvTranspose1d, (32, 2),
                                     (192, 16, 50), (16, 32, 2), (32,))

    @suppress_warnings
    def test_lazy_conv_transpose1d_state(self):
        self._check_lazy_conv_state(lambda: nn.ConvTranspose1d(16, 32, 2),
                                    lambda: nn.LazyConvTranspose1d(32, 2),
                                    (16, 32, 2), (32,))

    @suppress_warnings
    def test_lazy_conv_transpose2d(self):
        self._check_lazy_conv(nn.ConvTranspose2d, nn.LazyConvTranspose2d, torch.nn.functional.conv_transpose2d,
                              (32, 2), (192, 16, 8, 6), (16, 32, 2, 2), (32,))

    @suppress_warnings
    def test_lazy_conv_transpose2d_pickle(self):
        self._check_lazy_conv_pickle(nn.ConvTranspose2d, nn.LazyConvTranspose2d, (32, 2),
                                     (192, 16, 8, 6), (16, 32, 2, 2), (32,))

    @suppress_warnings
    def test_lazy_conv_transpose2d_state(self):
        self._check_lazy_conv_state(lambda: nn.ConvTranspose2d(16, 32, 2),
                                    lambda: nn.LazyConvTranspose2d(32, 2),
                                    (16, 32, 2, 2), (32,))

    @suppress_warnings
    def test_lazy_conv_transpose3d(self):
        self._check_lazy_conv(nn.ConvTranspose3d, nn.LazyConvTranspose3d, torch.nn.functional.conv_transpose3d,
                              (32, 2), (192, 16, 8, 7, 6), (16, 32, 2, 2, 2), (32,))

    @suppress_warnings
    def test_lazy_conv_transpose3d_pickle(self):
        self._check_lazy_conv_pickle(nn.ConvTranspose3d, nn.LazyConvTranspose3d, (32, 2),
                                     (192, 16, 8, 7, 6), (16, 32, 2, 2, 2), (32,))

    @suppress_warnings
    def test_lazy_conv_transpose3d_state(self):
        self._check_lazy_conv_state(lambda: nn.ConvTranspose3d(16, 32, 2),
                                    lambda: nn.LazyConvTranspose3d(32, 2),
                                    (16, 32, 2, 2, 2), (32,))

    def _check_lazy_batchnorm(self, cls, lazy_cls, input_shape):
        for affine in [False, True]:
            for track_running_stats in [False, True]:
                lazy_module = lazy_cls(affine=affine, track_running_stats=track_running_stats)

                if affine:
                    self.assertIsInstance(lazy_module.weight, UninitializedParameter)
                    self.assertIsInstance(lazy_module.bias, UninitializedParameter)
                if track_running_stats:
                    self.assertIsInstance(lazy_module.running_mean, UninitializedBuffer)
                    self.assertIsInstance(lazy_module.running_var, UninitializedBuffer)

                input = torch.ones(*input_shape)
                y = lazy_module(input)
                self.assertIsInstance(lazy_module, cls)
                self.assertNotIsInstance(lazy_module, lazy_cls)

                num_features = input_shape[1]
                module = cls(num_features, affine=affine, track_running_stats=track_running_stats)
                expected = module(input)

                if module.weight is not None:
                    self.assertEqual(lazy_module.weight.shape, module.weight.shape)
                    self.assertEqual(lazy_module.weight, module.weight)
                if module.bias is not None:
                    self.assertEqual(lazy_module.bias.shape, module.bias.shape)
                    self.assertEqual(lazy_module.bias, module.bias)
                if module.running_mean is not None:
                    self.assertEqual(lazy_module.running_mean.shape, module.running_mean.shape)
                    self.assertEqual(lazy_module.running_mean, module.running_mean)
                if module.running_var is not None:
                    self.assertEqual(lazy_module.running_var.shape, module.running_var.shape)
                    self.assertEqual(lazy_module.running_var, module.running_var)
                if module.num_batches_tracked is not None:
                    self.assertEqual(lazy_module.num_batches_tracked.shape, module.num_batches_tracked.shape)
                    self.assertEqual(lazy_module.num_batches_tracked, module.num_batches_tracked)

    def _check_lazy_batchnorm_pickle(self, cls, lazy_cls, input_shape):
        for affine in [False, True]:
            for track_running_stats in [False, True]:
                module = lazy_cls(affine=affine, track_running_stats=track_running_stats)
                module = pickle.loads(pickle.dumps(module))

                self.assertIsInstance(module, lazy_cls)
                if affine:
                    self.assertIsInstance(module.weight, UninitializedParameter)
                    self.assertIsInstance(module.bias, UninitializedParameter)
                if track_running_stats:
                    self.assertIsInstance(module.running_mean, UninitializedBuffer)
                    self.assertIsInstance(module.running_var, UninitializedBuffer)

                input = torch.ones(*input_shape)
                module(input)  # fully materialized
                module = pickle.loads(pickle.dumps(module))

                self.assertNotIsInstance(module, lazy_cls)
                self.assertIsInstance(module, cls)
                if affine:
                    self.assertNotIsInstance(module.weight, UninitializedParameter)
                    self.assertNotIsInstance(module.bias, UninitializedParameter)
                if track_running_stats:
                    self.assertNotIsInstance(module.running_mean, UninitializedBuffer)
                    self.assertNotIsInstance(module.running_var, UninitializedBuffer)

    def _check_lazy_batchnorm_state(self, cls, lazy_cls):
        module = cls(10)
        lazy_module = lazy_cls(affine=True, track_running_stats=True)
        lazy_module.load_state_dict(module.state_dict())
        # Parameters have been initialized but the module won't become a full
        # Conv one until the first iteration. This is due to
        # limitations on the state_dict loading logic
        self.assertFalse(lazy_module.has_uninitialized_params())
        self.assertEqual(lazy_module.weight.shape, (10,))
        self.assertEqual(lazy_module.bias.shape, (10,))
        self.assertEqual(lazy_module.running_mean.shape, (10,))
        self.assertEqual(lazy_module.running_var.shape, (10,))

        module = cls(10)
        lazy_module = lazy_cls()
        with self.assertRaisesRegex(RuntimeError, 'shape of an uninitialized'):
            module.load_state_dict(lazy_module.state_dict())

    def test_lazy_batchnorm1d(self):
        self._check_lazy_batchnorm(nn.BatchNorm1d, nn.LazyBatchNorm1d, (16, 3, 6))
        self._check_lazy_batchnorm(nn.BatchNorm1d, nn.LazyBatchNorm1d, (16, 6))

    def test_lazy_batchnorm1d_pickle(self):
        self._check_lazy_batchnorm_pickle(nn.BatchNorm1d, nn.LazyBatchNorm1d, (16, 3, 6))
        self._check_lazy_batchnorm_pickle(nn.BatchNorm1d, nn.LazyBatchNorm1d, (16, 6))

    def test_lazy_batchnorm1d_state(self):
        self._check_lazy_batchnorm_state(nn.BatchNorm1d, nn.LazyBatchNorm1d)
        self._check_lazy_batchnorm_state(nn.BatchNorm1d, nn.LazyBatchNorm1d)

    def test_lazy_batchnorm2d(self):
        self._check_lazy_batchnorm(nn.BatchNorm2d, nn.LazyBatchNorm2d, (16, 3, 6, 7))

    def test_lazy_batchnorm2d_pickle(self):
        self._check_lazy_batchnorm_pickle(nn.BatchNorm2d, nn.LazyBatchNorm2d, (16, 3, 6, 7))

    def test_lazy_batchnorm2d_state(self):
        self._check_lazy_batchnorm_state(nn.BatchNorm2d, nn.LazyBatchNorm2d)
        self._check_lazy_batchnorm_state(nn.BatchNorm2d, nn.LazyBatchNorm2d)

    def test_lazy_batchnorm3d(self):
        self._check_lazy_batchnorm(nn.BatchNorm3d, nn.LazyBatchNorm3d, (16, 3, 6, 7, 8))

    def test_lazy_batchnorm3d_pickle(self):
        self._check_lazy_batchnorm_pickle(nn.BatchNorm3d, nn.LazyBatchNorm3d, (16, 3, 6, 7, 8))

    def test_lazy_batchnorm3d_state(self):
        self._check_lazy_batchnorm_state(nn.BatchNorm3d, nn.LazyBatchNorm3d)
        self._check_lazy_batchnorm_state(nn.BatchNorm3d, nn.LazyBatchNorm3d)

    @suppress_warnings
    def test_materialize_dtype(self):
        module = LazyModule()
        module.register_parameter('test_param', UninitializedParameter())
        module.test_param.materialize(10)
        self.assertTrue(module.test_param.dtype == torch.float64)
        module = LazyModule()
        module.register_parameter('test_param', UninitializedParameter())
        module.half()
        module.test_param.materialize(10)
        self.assertTrue(module.test_param.dtype == torch.float16)

    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    @suppress_warnings
    def test_materialize_device(self):
        module = LazyModule()
        module.register_parameter('test_param', UninitializedParameter())
        module.test_param.materialize(10)
        self.assertTrue(module.test_param.device.type == 'cpu')
        module = LazyModule()
        module.register_parameter('test_param', UninitializedParameter())
        module.cuda()
        module.test_param.materialize(10)
        self.assertTrue(module.test_param.device.type == 'cuda')

    @suppress_warnings
    def test_chained_initialization(self):
        class MyNetwork(torch.nn.Module):
            def __init__(self):
                super(MyNetwork, self).__init__()
                self.linear_1 = torch.nn.LazyLinear(15)
                self.linear_2 = torch.nn.LazyLinear(10)

            def forward(self, x):
                y = self.linear_1(x)
                return self.linear_2(y)

        net = MyNetwork()
        net(torch.ones(5, 10))
        self.assertTrue(net.linear_1.weight.shape == (15, 10))
        self.assertTrue(net.linear_1.bias.shape == (15,))
        self.assertTrue(net.linear_2.weight.shape == (10, 15))
        self.assertTrue(net.linear_2.bias.shape == (10,))

    @suppress_warnings
    def test_optimizer_pass(self):
        optimizers = [torch.optim.Adadelta, torch.optim.Adagrad, torch.optim.Adam,
                      torch.optim.AdamW, torch.optim.Adamax,
                      torch.optim.ASGD, torch.optim.SGD, torch.optim.Rprop,
                      torch.optim.RMSprop, torch.optim.LBFGS]

        def run_step(module, optim):
            self.assertIsInstance(optim.param_groups[0]['params'][0], UninitializedParameter)
            module.test_param.materialize(10)
            self.assertIsInstance(optim.param_groups[0]['params'][0], Parameter)
            self.assertNotIsInstance(optim.param_groups[0]['params'][0], UninitializedParameter)
            for p in module.parameters():
                p.grad = torch.rand_like(p)
            if isinstance(optim, torch.optim.LBFGS):
                optim.step(lambda: 1.0)
            else:
                optim.step()

        for optim_cls in optimizers:
            module = LazyModule()
            module.register_parameter('test_param', UninitializedParameter())
            if optim_cls is torch.optim.SGD:
                optim = optim_cls(module.parameters(), lr=0.0)
            elif optim_cls is torch.optim.Adagrad:
                with self.assertRaisesRegex(ValueError, 'uninitialized parameter'):
                    optim = optim_cls(module.parameters())
                continue
            else:
                optim = optim_cls(module.parameters())
            run_step(module, optim)

    @suppress_warnings
    def test_weight_norm(self):
        m = nn.LazyLinear(7)
        with self.assertRaisesRegex(ValueError, 'have uninitialized parameters.'):
            m = torch.nn.utils.weight_norm(m)

    @suppress_warnings
    def test_spectral_norm(self):
        m = nn.LazyLinear(7)
        with self.assertRaisesRegex(ValueError, 'have uninitialized parameters.'):
            m = torch.nn.utils.spectral_norm(m)

    @suppress_warnings
    def test_invalid_functions(self):
        param = torch.nn.parameter.UninitializedParameter()
        with self.assertRaisesRegex(ValueError, 'uninitialized parameter'):
            torch.empty_like(param)

        with self.assertRaisesRegex(ValueError, 'uninitialized parameter'):
            torch.add(param, param)

        with self.assertRaisesRegex(ValueError, 'uninitialized parameter'):
            param + param

class TestFunctionalPickle(TestCase):

    # issue gh-38137
    def test_pickle_softsign(self):
        # Make sure it does not throw an exception
        s = pickle.dumps(F.softsign)
if __name__ == '__main__':
    run_tests()
