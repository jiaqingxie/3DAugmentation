from ogb.lsc import PCQM4Mv2Dataset
from ogb.utils import smiles2graph
from rdkit import Chem

dataset=PCQM4Mv2Dataset(only_smiles=True)
i=1
m=Chem.MolFromSmiles(dataset[i][0])
m = Chem.AddHs(m)
atomic_symbols = []
xyz_coordinates = []

with open("dataset/pcqm4m-v2_xyz/00000000_00009999/1.xyz", "r") as file:
    for line_number,line in enumerate(file):
        if line_number == 0:
            num_atoms = int(line)
        elif line_number == 1:
            comment = line # might have useful information
        else:
            atomic_symbol, x, y, z = line.split()
            atomic_symbols.append(atomic_symbol)
            xyz_coordinates.append([float(x),float(y),float(z)])
print(dataset[i][0])
print(atomic_symbols)
atomic_symbols=atomic_symbols[0:17]
xyz_coordinates=xyz_coordinates[0:17]
from rdkit.Geometry import Point3D
# import ipdb
# ipdb.set_trace()

# conf = m.GetConformer()
# import ipdb
# ipdb.set_trace()
# # in principal, you should check that the atoms match
# for i in range(m.GetNumAtoms()):
#    x,y,z = xyz_coordinates[i]
#    conf.SetAtomPosition(i,Point3D(x,y,z))

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
x=[]
y=[]
z=[]
for i in range(len(xyz_coordinates)):
    x1,y1,z1=xyz_coordinates[i]
    x.append(x1)
    y.append(y1)
    z.append(z1)

# Make data

# Plot the surface
#ax.plot_surface(x, y, z, color='b')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#定义坐标轴
fig = plt.figure()
ax1 = plt.axes(projection='3d')
##################我是分割线#####################
import numpy as np
for i in range(17):

    ax1.scatter3D(x[i],y[i],z[i], cmap='b')
    # ax1.scatter3D(x[13],y[13],z[13], cmap='r')
    # ax1.scatter3D(x[14:],y[14:],z[14:], cmap='y')  #绘制散点图   #绘制空间曲线
    ax1.text(x[i],y[i],z[i],  '%s' % (str(i)), size=20, zorder=1,  
        color='k') 
plt.show()
plt.savefig("3d.png")