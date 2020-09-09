import sherman_morrison_python_functions as sm
import numpy as np
import pickle
fid = open('badDictionary.pkl','rb')
Df = pickle.load(fid)
R = pickle.load(fid)
Q = pickle.load(fid)
Gf = pickle.load(fid)
fid.close()
import sporco.plot
import sporco.util
import sporco.linalg
Df.shape
Gf.shape
D = sporco.linalg.ifftn(Df,(78,78),(0,1))
G = sporco.linalg.ifftn(Gf,(78,78),(0,1))
D.shape
np.max(np.abs(D[slice(8,78),slice(8,78)]))
np.max(np.abs(D))
Dr = D[slice(0,8),slice(0,8)]
Dr.shape
Gr = G[slice(0,8),slice(0,8)]
Gr.shape
sporco.plot.imview(sporco.util.tiledict(Dr.squeeze()))
sporco.plot.imview(sporco.util.tiledict(Gr.squeeze()))
np.amax(np.abs(Gr))
np.amax(np.abs(Dr))
