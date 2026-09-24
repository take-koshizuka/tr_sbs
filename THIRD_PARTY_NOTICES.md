# Third-party notices

## Alanine-dipeptide coordinate-transform reference

The minimum-energy alanine-dipeptide coordinates embedded in
`src/tr_asbs/fab/aldp_transform_data.py` are derived from
[`juliusberner/sde_sampler`](https://github.com/juliusberner/sde_sampler) at
commit `06b553a3f4b0c592d6429fa2c309d47552accbfd`.

Copyright (c) 2023 Julius Berner and Lorenz Richter

The original data is distributed under the MIT License:

> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

## Variational sampler ports

The PyTorch Path Integral Sampler (PIS) and Time-Reversed Diffusion Sampler
(DIS) implementations in `src/tr_asbs/variational` are adapted from the
corresponding implementations in
[`DenisBless/variational_sampling_methods`](https://github.com/DenisBless/variational_sampling_methods)
at commit `32de62189abe07e96742735d91adbb959692d8c3`.

Copyright (c) 2023 Denis Blessing

The original implementation is distributed under the MIT License. Its license
notice is reproduced below:

> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

The implementation is based on:

- Zhang and Chen, [“Path Integral Sampler: a stochastic control approach for
  sampling”](https://arxiv.org/abs/2111.15141), 2021.
- Berner, Richter, and Ullrich, [“An optimal control perspective on
  diffusion-based generative modeling”](https://arxiv.org/abs/2211.01364),
  TMLR 2024.
- Blessing et al., [“Beyond ELBOs: A Large-Scale Evaluation of Variational
  Methods for Sampling”](https://arxiv.org/abs/2406.07423), ICML 2024.
