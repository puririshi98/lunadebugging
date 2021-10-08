import torch
import torch.nn.functional as F
import torch.nn as nn
import itertools
import math
nan=float('nan')
inf=float('inf')
def test_grid_sample_large_index_2d(device, dtype):
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
    assert sum(i * s for i, s in zip(large_view.size(), large_view.stride())) >= 2 ** 31, "View must use 64-bit indexing"
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

        assert torch.equal(a, b)
        assert torch.equal(small_image.grad, large_view.grad)

        small_image.grad.zero_()
        large_view.grad.zero_()

def test_grid_sample_large_index_3d(device, dtype):
	# Test 64-bit indexing with grid_sample (gh-41656)
	# Try accessing the corners, there should be no segfault
	coords = torch.full((1, 2, 2, 2, 3), 1., device=device, dtype=dtype)
	im = torch.zeros([1, 1, 2, 32769, 32768], device=device, dtype=dtype)

	result = F.grid_sample(im, coords, align_corners=False)
	assert torch.equal(result,torch.zeros((1, 1, 2, 2, 2), device=device, dtype=dtype))

	# Compare sampling with large strides to the same op on a contiguous tensor
	coords = torch.rand(1, 1, 4, 4, 3, device=device, dtype=dtype)
	large_view = im[..., 127::128]
	small_image = torch.rand_like(large_view)
	large_view[...] = small_image
	small_image.requires_grad, large_view.requires_grad = True, True
	assert sum(i * s for i, s in zip(large_view.size(), large_view.stride())) >= 2 ** 31
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

def test_fractional_max_pool_nan_inf(device, dtype):
    for num_dim in [2, 3]:
        fn_name = 'FractionalMaxPool{}d'.format(num_dim)
        fn = getattr(nn, fn_name)(kernel_size=2, output_size=1)
        x = torch.full([1, 1] + num_dim * [3], nan, device=device, dtype=dtype, requires_grad=True)
        res = fn(x)
        res.backward(torch.randn_like(res))
        assert (math.isnan(res.item()))

        x2 = torch.full([1, 1] + num_dim * [3], -inf, device=device, dtype=dtype, requires_grad=True)
        res2 = fn(x2)
        res2.backward(torch.randn_like(res2))
        assert (math.isinf(res2.item()))
test_fractional_max_pool_nan_inf('cuda:0', torch.double)
print(torch.cuda.memory_summary())
test_grid_sample_large_index_2d('cuda:0',torch.float)
print(torch.cuda.memory_summary())
test_grid_sample_large_index_3d('cuda:0',torch.float)
