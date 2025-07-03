import os
from setuptools import setup, find_packages

import torch
from torch.utils.cpp_extension import BuildExtension


ext_modules = []
if torch.cuda.is_available():
    from torch.utils.cpp_extension import CUDAExtension
    seqlen_dynamic_attention_ext = CUDAExtension(
        name='my_softmax.cuda',
        sources=[
            os.path.join('csrc', 'my_softmax.cpp'),
            os.path.join('csrc', 'naive_softmax.cu'),
            os.path.join('csrc', 'better_softmax.cu'),
        ],
        extra_compile_args=['-std=c++17', '-O3'],
    )
    ext_modules.append(seqlen_dynamic_attention_ext)
else:
    raise RuntimeError('CUDA is not available.')


setup(
    name='MySoftmax',
    description='Simple CUDA Softmax Kernel',
    packages=find_packages(),
    install_requires=['torch'],
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
)
