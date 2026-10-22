# Third-party notices

## Vendored MACE source

`src/mace/` is derived from [ACEsuit/mace](https://github.com/ACEsuit/mace).
Its retained version string is `0.3.15`; the vendored tree contains local
modifications and is not a byte-identical upstream release.

Copyright (c) 2022 ACEsuit/mace

The upstream MIT notice from
[`v0.3.15/LICENSE.md`](https://github.com/ACEsuit/mace/blob/a2803e5e9eef71ee047d25addf584415ec0c0ca3/LICENSE.md),
tag commit `a2803e5e9eef71ee047d25addf584415ec0c0ca3`, is copied verbatim to
[`src/mace/MIT.md`](src/mace/MIT.md). This supplies the notice referenced by
the existing MACE source headers. File-level and nested notices are retained.

MACE contains components with their own attribution and license terms:

| Vendored component | Attribution and provenance | Included license |
| --- | --- | --- |
| `src/mace/tools/torch_geometric/` | The [retained README](src/mace/tools/torch_geometric/README.md) identifies a modified PyTorch Geometric subset, copied through NequIP, and cites PyG 2.0.1. PyG: Copyright (c) 2021 Matthias Fey, Jiaxuan You. NequIP: Copyright (c) 2021 The President and Fellows of Harvard College. | [PyG MIT](licenses/PyG-2.0.1-MIT.txt), [NequIP MIT](licenses/NequIP-MIT.txt) |
| `src/mace/tools/scatter.py` | Its header credits `mir-group/pytorch_runstats` and `rusty1s/pytorch_scatter`. Copyright (c) 2021 MIR@Harvard; Copyright (c) 2020 Matthias Fey. | [pytorch-runstats MIT](licenses/pytorch-runstats-MIT.txt), [torch-scatter MIT](licenses/torch-scatter-MIT.txt) |
| `src/mace/tools/cg_cueq_tools.py` | Its header identifies modifications from NVIDIA cuEquivariance. Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0. | [Apache License 2.0](licenses/Apache-2.0.txt) |
| `src/mace/tools/fairchem_dataset/lmdb_dataset_tools.py` | Its [retained notice](src/mace/tools/fairchem_dataset/fairchem_readme.md) identifies FairChem-derived code (Copyright (c) Meta, Inc. and its affiliates) and an ASE-derived LMDBDatabase component. The latter is identified there as LGPL 2.1. | [FairChem MIT](licenses/FairChem-MIT.txt), [GNU LGPL 2.1](licenses/LGPL-2.1.txt) |

The graph subset is identified as MIT-licensed by its upstream projects;
the Apache-2.0 header applies to the cuEquivariance-derived file. Original
component commits are not recorded in the graph/scatter subset headers, so the
following upstream versions document the included notices without asserting
an exact import revision:

- [PyTorch Geometric 2.0.1 license](https://github.com/pyg-team/pytorch_geometric/blob/2.0.1/LICENSE).
- [NequIP v0.5.6 license](https://github.com/mir-group/nequip/blob/v0.5.6/LICENSE).
- [torch-scatter 2.0.9 license](https://github.com/rusty1s/pytorch_scatter/blob/2.0.9/LICENSE).
- [pytorch-runstats license](https://github.com/mir-group/pytorch_runstats/blob/main/LICENSE).
- [FairChem license](https://github.com/FAIR-Chem/fairchem/blob/main/LICENSE.md).
- [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt).
- [GNU LGPL 2.1](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt),
  corresponding to the [ASE license source](https://gitlab.com/ase/ase/-/blob/master/LICENSE)
  linked by the retained component notice. The included text is copied from
  the [SPDX license-text distribution](https://github.com/spdx/license-list-data/blob/main/text/LGPL-2.1-only.txt).

Download locations and file checksums are recorded in
[`licenses/SOURCES.md`](licenses/SOURCES.md).

These source-code notices do not relicense separately downloaded pretrained
weights or datasets. MACE foundation-model loaders retain their model-specific
license messages.

## Historical alanine-dipeptide coordinate-transform reference

The following attribution is retained from an earlier coordinate-transform
implementation. The named file `src/tr_asbs/fab/aldp_transform_data.py` is not
present in the current source; the current IC pipeline loads external
trajectory data for calibration.

The retained historical notice identifies minimum-energy alanine-dipeptide
coordinates in that file as derived from
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
