# AI for Science 工具集使用指南

这个Docker环境预装了丰富的AI for Science工具，涵盖了化学、物理、生物、材料科学等多个领域。

## 🧪 化学信息学

### RDKit - 化学信息学工具包
```python
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
from rdkit.Chem.Draw import IPythonConsole

# 创建分子对象
mol = Chem.MolFromSmiles('CCO')  # 乙醇
print(f"分子量: {Descriptors.MolWt(mol):.2f}")
print(f"LogP: {Descriptors.MolLogP(mol):.2f}")

# 在Jupyter中显示分子结构
Draw.MolToImage(mol)
```

### DeepChem - 深度学习化学
```python
import deepchem as dc
import numpy as np

# 加载数据集
tasks, datasets, transformers = dc.molnet.load_tox21()
train_dataset, valid_dataset, test_dataset = datasets

# 创建图卷积模型
model = dc.models.GraphConvModel(n_tasks=len(tasks), mode='classification')
model.fit(train_dataset, nb_epoch=10)
```

## 🔬 物理和材料科学

### ASE (Atomic Simulation Environment)
```python
from ase import Atoms
from ase.build import bulk
from ase.visualize import view

# 创建晶体结构
cu = bulk('Cu', 'fcc', a=3.6)
print(f"原子数: {len(cu)}")
print(f"晶胞体积: {cu.get_volume():.2f} Ų")

# 可视化 (在支持X11的环境中)
# view(cu)
```

### Pymatgen - 材料分析
```python
from pymatgen.core import Structure, Lattice
from pymatgen.analysis.structure_analyzer import VoronoiConnectivity

# 创建简单立方结构
lattice = Lattice.cubic(4.0)
structure = Structure(lattice, ["Li", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
print(f"化学式: {structure.composition}")
```

## 🧬 生物信息学

### BioPython - 生物序列分析
```python
from Bio.Seq import Seq
from Bio.SeqUtils import GC
from Bio import SeqIO

# DNA序列操作
dna_seq = Seq("AGTACACTGGT")
print(f"DNA: {dna_seq}")
print(f"转录: {dna_seq.transcribe()}")
print(f"翻译: {dna_seq.translate()}")
print(f"GC含量: {GC(dna_seq):.1f}%")
```

### Scanpy - 单细胞分析
```python
import scanpy as sc
import anndata as ad
import pandas as pd

# 设置scanpy
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=80, facecolor='white')

# 创建示例数据
adata = sc.datasets.pbmc68k_reduced()
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.pl.umap(adata, color='bulk_labels')
```

## 📊 高级可视化

### py3Dmol - 分子3D可视化
```python
import py3Dmol

# 创建分子查看器
view = py3Dmol.view(width=400, height=300)
view.addModel('CCO', 'smiles')
view.setStyle({'stick': {}})
view.zoomTo()
view.show()
```

### Mayavi - 科学3D可视化
```python
from mayavi import mlab
import numpy as np

# 创建3D数据
x, y, z = np.mgrid[-10:10:20j, -10:10:20j, -10:10:20j]
s = np.sin(x*y*z) / (x*y*z)

# 3D等值面
mlab.contour3d(s)
mlab.show()
```

## 🔢 科学计算

### JAX - 高性能数值计算
```python
import jax.numpy as jnp
from jax import grad, jit, vmap
import jax.random as random

# 定义函数
@jit
def predict(params, inputs):
    return jnp.dot(inputs, params)

# 自动求导
grad_predict = grad(predict)

# 向量化
batch_predict = vmap(predict, in_axes=(None, 0))
```

### SymPy - 符号计算
```python
import sympy as sp

# 定义符号
x, y = sp.symbols('x y')
expr = x**2 + 2*x*y + y**2

# 符号运算
simplified = sp.simplify(expr)
derivative = sp.diff(expr, x)
integral = sp.integrate(expr, x)

print(f"表达式: {expr}")
print(f"简化: {simplified}")
print(f"对x求导: {derivative}")
```

## 🚀 图神经网络

### PyTorch Geometric
```python
import torch
import torch_geometric
from torch_geometric.data import Data

# 创建图数据
edge_index = torch.tensor([[0, 1, 1, 2],
                          [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)
print(f"节点数: {data.num_nodes}")
print(f"边数: {data.num_edges}")
```

### DGL (Deep Graph Library)
```python
import dgl
import torch

# 创建图
g = dgl.graph(([0, 0, 1, 5], [1, 2, 2, 0]))
g.ndata['feat'] = torch.randn(6, 10)
print(f"图信息: {g}")
```

## 🔧 优化工具

### CVXPY - 凸优化
```python
import cvxpy as cp
import numpy as np

# 定义优化变量
x = cp.Variable(2)

# 定义目标函数和约束
objective = cp.Minimize(cp.sum_squares(x))
constraints = [x >= 0, cp.sum(x) == 1]

# 求解问题
prob = cp.Problem(objective, constraints)
prob.solve()

print(f"最优值: {prob.value}")
print(f"最优解: {x.value}")
```

### Ray - 分布式计算
```python
import ray
import time

@ray.remote
def slow_function(i):
    time.sleep(1)
    return i

# 初始化Ray
ray.init()

# 并行执行
start = time.time()
results = ray.get([slow_function.remote(i) for i in range(4)])
end = time.time()

print(f"并行执行时间: {end - start:.2f}秒")
ray.shutdown()
```

## 📈 专业分析工具

### 分子动力学分析 (MDTraj)
```python
import mdtraj as md
import numpy as np

# 模拟轨迹分析示例
# traj = md.load('trajectory.xtc', top='topology.pdb')
# distances = md.compute_distances(traj, [[0, 1]])
```

### 统计建模 (Statsmodels)
```python
import statsmodels.api as sm
import numpy as np

# 生成数据
np.random.seed(0)
X = np.random.randn(100, 2)
y = X[:, 0] + 2*X[:, 1] + np.random.randn(100)

# 线性回归
X = sm.add_constant(X)
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())
```

## 💡 使用建议

1. **环境管理**: 使用conda环境管理不同项目的依赖
2. **GPU加速**: 大多数深度学习库都支持CUDA加速
3. **可视化**: 在Jupyter中使用`%matplotlib inline`或`%matplotlib widget`
4. **内存管理**: 处理大数据时注意内存使用
5. **并行计算**: 利用Ray或Dask进行大规模计算

## 🔗 相关资源

- [RDKit文档](https://rdkit.readthedocs.io/)
- [DeepChem教程](https://deepchem.readthedocs.io/)
- [ASE教程](https://wiki.fysik.dtu.dk/ase/)
- [Pymatgen文档](https://pymatgen.org/)
- [BioPython教程](https://biopython.org/wiki/Documentation)
- [Scanpy教程](https://scanpy.readthedocs.io/)
- [JAX文档](https://jax.readthedocs.io/)
- [PyTorch Geometric教程](https://pytorch-geometric.readthedocs.io/)

这个环境为AI for Science研究提供了完整的工具链，可以支持从数据预处理到模型训练、从分子设计到材料发现的全流程研究工作。 