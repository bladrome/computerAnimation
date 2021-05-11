import numpy as np
import networkx as nx
import trimesh
mesh = trimesh.load("Trex.obj")

# Get the 'tail' by coor z
tailindexlist = [index for index in np.argsort(mesh.vertices[:,2])[:45]]
tailindexorder = {index:i for i, index in enumerate(tailindexlist)}
# Get the reference of fixpoint
anchorindex  =  tailindexlist[-11:]
handleindex  =  tailindexlist[:14]

# Fix point
anchorposition = mesh.vertices[anchorindex]
handleposition = mesh.vertices[handleindex] + [-20, 0, 0]

# Construct Adjacent Matrix of Edit Points
A = nx.Graph()
A.add_edges_from(mesh.edges)
A = nx.to_numpy_array(A, nodelist=tailindexlist)

# Laplacian Matrix and Delta Matrix
L = np.eye(A.shape[0]) - np.matmul(np.linalg.pinv(np.diag(A.sum(axis=0))), A)
Delta = np.matmul(L, mesh.vertices[tailindexlist])

# Augment I
colnameindex = anchorindex
colnameindex.extend(handleindex)
augI = np.zeros(shape=(anchorposition.shape[0] + handleposition.shape[0], L.shape[1]))
ijindex = [(i, tailindexorder[col]) for i,col in enumerate(colnameindex)]
for i in ijindex: augI[i] = 1
augL = np.row_stack((L, augI))

# Augment Delta (Fix points)
augDelta = np.row_stack((Delta,
                         anchorposition,
                         handleposition))

# Solve
V =  np.matmul(np.matmul(np.linalg.pinv(np.matmul(augL.T,
                                                  augL)),
                         augL.T),
               augDelta)

# Edit
mesh.vertices[tailindexlist] = V
mesh.export("newTrex.obj")
