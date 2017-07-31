import numpy as np
from collections import Counter
from ovito.io import *
from ovito.modifiers import *

node = import_file("dump.atom", multiple_frames=True)

voro = VoronoiAnalysisModifier(
    compute_indices = True,
    use_radii = True,
    edge_count = 10, # Length after which Voronoi index vectors are truncated
    edge_threshold = 0.0
)
node.modifiers.append(voro)

# voroIndex = node.output['Voronoi Index'].array

# Define the custom modifier function:
def  voroIndices(frame, input, output):

    # Access the per-particle displacement magnitudes computed by an existing 
    # Displacement Vectors modifier that precedes this custom modifier in the 
    # data pipeline:
    voroTable = input.particle_properties['Voronoi Index'].array
    voroTuples = [tuple(e) for e in voroTable]
    # Output results
    d=Counter(voroTuples)
    for item, value in d.items():
    	# print(frame,str(item), value)
    	output.attributes[str(item)] = value



# Insert custom modifier into the data pipeline.
node.modifiers.append(PythonScriptModifier(function = voroIndices))
print("==> The number of frames is",node.source.num_frames)

print ("==> Computing the concentration series from the Voronoi signatures...")
result={}
for frame in range(node.source.num_frames):
	node.compute(frame)	
	# called={k:0 for k in result.keys()}
	for name in node.output.attribute_names:
		# called[name]=1
		try:
			result[name].append( [frame,node.output.attributes[name]] )
		except Exception:
			result[name]=[]
			result[name].append( [frame,node.output.attributes[name]] )

# Now, reconstruct time series filling the zeros
frames=np.arange(node.source.num_frames)

for name in result.keys():
	result[name]=np.array(result[name])
	z=np.zeros(len(frames))
	# if len (result[name])==2:
	z[result[name][:,0]]=result[name][:,1]
	result[name]=np.array([frames ,z])


print ("==> Computing the covariance matrix ...")
structMat = []

for k in result.keys():
	if k[0]=='(':
		structMat.append(result[k][1])

structMat=np.array(structMat)

covMat = np.cov(structMat)
import pylab as pl
normed=covMat/np.min(np.abs(covMat))
I=np.log10(np.abs(normed))*np.sign(normed)
pl.matshow(I, cmap=pl.cm.magma )
pl.colorbar()
pl.savefig('mat.png')
