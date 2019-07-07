# nxMultilayerNet
Multilayer networks implemented on top of NetworkX. 


## Installation

This package supports installation with setuptools. The easiest way to 
install the latest version of this package and its dependencies is to use 
```
pip install git+https://github.com/LJeub/nxMultilayerNet.git@master
```
To install a particular version, replace `master` above by the appropriate commit
identifier.

This package makes 
extensive use of features introduced in NetworkX-v2.3 and thus does not
work with earlier versions. 


## Basics

*This README only explains a subset of the functionality and is incomplete.
For more information look at the source code and inline documentation.*

This package defines two classes for working with multilayer networks 
`MultilayerGraph` and `MultilayerDiGraph` which extend NetworkX `Graph`
and `DiGraph` respectively. These classes work by storing the flattened
network corresponding to the multilayer network as a NetworkX `Graph` or 
`DiGraph` and existing NetworkX methods can thus be used to interact with
the flattened network. Nodes are always tuples of a fixed length, i.e. 
`node=(n, a_1,...,a_d)` where `node[0]` is the label of the physical node
and `node[i]`, `i>0` are the labels corresponding to the different aspects.

The examples below assume that the package is imported as
```python
import nxmultilayer as nxm
```

### Construct network

Construct empty undirected multilayer network with a single aspect:
```python
mg = nxm.MultilayerGraph(n_aspects=1)
```

Construct by copying `g` (an existing multilayer network or NetworkX graph or similar,
provided the nodes have the correct format):
```python
mg = nxm.MultilayerGraph(g, n_aspects=d)
```
If `g` is a `MultilayerGraph` or `MultilayerDiGraph`, `n_aspects` is ignored,
otherwise `d` needs to specify the number of aspects of `g`.


### Layer views

Intralayer edges can be accessed through the `layers` attribute, i.e.:
```python
layer_a = mg.layers[a]
```
where `a` is a tuple of layer indeces, i.e. `a=(a_1,...,a_d)`. Use
```python
all_interlayers = mg.layers[:]
```
to get all intralayer edges.


