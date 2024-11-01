from fista.kernels.mm_sub import matmul_sub
from fista.utils import set_seed
import torch
import unittest

class TestMMSub(unittest.TestCase):
    def setUp(self):
        set_seed(0)
        self.a_ = torch.randn((512, 768), dtype=torch.float32, device='cuda')
        self.b_ = torch.randn((768, 1024), dtype=torch.float32, device='cuda')
        self.c_ = torch.randn((512, 1024), dtype=torch.float32, device='cuda')

    def test_ieee(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        o = matmul_sub(self.a_, self.b_, self.c_, input_precision="ieee")
        r = self.c_ - self.a_ @ self.b_
        assert torch.allclose(r, o, atol=1e-4)

    @unittest.skip
    def test_tf32(self):
        # cannot match tf32 torch backend with triton tl.dot input precision
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        o = matmul_sub(self.a_, self.b_, self.c_, input_precision="tf32")
        r = self.c_ - self.a_ @ self.b_
        assert torch.allclose(r, o, atol=1e-4)


if __name__ == '__main__':
    unittest.main()