# chemprop_polaris_pyscf_demo

To celebrate the release of Chemprop version 2 I've put together this demo for integrating Chemprop with other packages (PySCF and Polaris) in Python code.

This was prohibitively difficult with Chemprop v1, but now is very straightforward.
I know from personal experience - in [`QuantumScents: Quantum-Mechanical Properties for 3.5k Olfactory Molecules`](https://pubs.acs.org/doi/10.1021/acs.jcim.3c01338) I used Hirshfeld charges as atomic features in Chemprop and it required a lot of hacky code.
This demo re-implements the same idea as that paper, but now in a few elegant Python functions and classes!

To run this demo, you'll need to install `'chemprop>=2.2.0' 'pyscf>=2.5.0'` (with Python 3.11 or 3.12) and this [hirshfeld analysis](https://github.com/frobnitzem/hirshfeld) PySCF extension.
You might also want to install the [GPU-accelerated version of PySCF](https://github.com/pyscf/gpu4pyscf?tab=readme-ov-file#installation) (pro tip: install `nvidia-cuda-toolkit`, if you don't already have it).

Feel free to use this code as a starting point for your own projects - it is licensed under the MIT license (see [LICENSE](./LICENSE)).
