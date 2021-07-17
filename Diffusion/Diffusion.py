import numpy as np
from matplotlib import pyplot as plt
import math
from qqdm import qqdm
from matplotlib import pyplot as plt


# Parameter setting
#=============================================

# global Parameter

iteration 	=	400
L 			=	1e-4								#cm
nodes		=	1600
delta_t		=	0.01									#s
delta_d		=	[1e-9, 5.333e-10, 7.086e-10]		#cm
NL			=	[0.007, 0.245, 0.459]				#%
NR			=	[0.001, 0.451, 0.99994]				#%
N0			=	[0.001, 0.240, 0.455, 0.99999]		#%
V 			=	[7.120, 8.590, 10.59, 16.12]		#cm^3

NvL	=	[NL[i]/V[i] for i in range(3)]
NvR	=	[NR[i]/V[i+1] for i in range(3)]
Nv0	=	[N0[i]/V[i] for i in range(4)]

def D_Coef_List(TemperatureData):
	DCoeff 	=	np.zeros((4,len(TemperatureData)))

	for n, Temperature in enumerate(TemperatureData):

		DCoeff[:,n]	=	[5.9e-5*np.exp(-138.8*1000/8.314/(Temperature+273))*1e8
					,3.65e-10*np.exp(-79.7*1000/8.314/(Temperature+273))*1e4
					,2.58e-8*np.exp(-85.2*1000/8.314/(Temperature+273))*1e4
					,3.98e-9*np.exp(-79.9*1000/8.314/(Temperature+273))*1e4]

	return DCoeff

def Diffusion_Coe(DiffusionCoefficient, Nvp_0, Nvp, Nvp_1):

	x = [Nvp_0, Nvp, Nvp_1]
	d = np.zeros(3)

	for i in range(3):
		if x[i] <= Nv_Sn(NvL[0]):
			d[i]	=	DiffusionCoefficient[0]
		elif x[i] > Nv_Sn(NvL[0]) and x[i] <= Nv_Sn(NvL[1]):
			d[i]	=	DiffusionCoefficient[1]
		elif x[i] > Nv_Sn(NvL[1]) and x[i] <= Nv_Sn(NvL[2]):
			d[i]	=	DiffusionCoefficient[2]
		elif x[i] > Nv_Sn(NvL[2]):
			d[i]	=	DiffusionCoefficient[3]


	d1 	=	2/((1/d[0])+1/d[1])
	d2 	=	2/((1/d[1])+1/d[2])	
	return d1, d2

# Grid setting
#=============================================

def CreateGrid():
	# Create a grid x: location, y: time, Nvsn

	#==========================================================================
	delta_x			=	L/(nodes - 1)	
	SimulationTime	=	delta_t*iteration
	print("Begin to create grid.")
	print("SimulationTime: {} s.\nNodes: {}.".format(SimulationTime,nodes))
	print('Delta t: {} s'.format(delta_t))
	print("================================================================")

	GridPoint		=	np.zeros((nodes, iteration, 3))

	xloc		=	-L/2

	for node in range(nodes):
		for step in range(iteration):
			GridPoint[node,step,:]	=	[xloc, step, 0]
		xloc	=	xloc+delta_x

	return	GridPoint

def initialModel():
	
	GridPoint	=	CreateGrid()

	GridPoint[0][:,-1]	=	Nv_Sn(Nv0[0])
	GridPoint[-1][:,-1]	=	Nv_Sn(Nv0[-1])

	for G in GridPoint:
		if G[0][0] <= 0:
			G[0][2]	=	Nv_Sn(Nv0[0])
		else:
			G[0][2]	=	Nv_Sn(Nv0[-1])

	return	GridPoint

def RuntIteration(Grid, TemData):
	
	print("Begin to run iteration.")
	print("================================================================")
	D_Coef		=	D_Coef_List(TemData)

	progressBar = qqdm(range(iteration))

	for step in progressBar:

		if step != 0:
			A_Matrix, B_Matrix	=	ComputeMatrix(D_Coef[:,step], Grid, step)
			Grid				=	UpdateGrid(A_Matrix, B_Matrix, Grid, step)

	print("Done !")
	return Grid

def UpdateGrid(A_Matrix, B_Matrix, Grid, step):

	B_Matrix[0,1]	=	B_Matrix[0,1]	-	A_Matrix[1,0]*Grid[0,step,-1]
	B_Matrix[0,-2]	=	B_Matrix[0,-2]	-	A_Matrix[-2,-1]*Grid[-1,step,-1]

	New_Nvp			=	np.dot(np.linalg.inv(A_Matrix[1:-1,1:-1]),B_Matrix[0,1:-1])
	Grid[1:-1,step,-1]	=	New_Nvp
	Grid[1,step,-1]		=	Grid[0,step,-1]
	Grid[-2,step,-1]	=	Grid[-1,step,-1]

	return Grid

# Compute functions
#=============================================

def ComputeMatrix(DiffusionCoefficient, Grid, step):

	delta_x			=	L/(nodes - 1)

	#	A Matrix
	A_Matrix	=	np.zeros((nodes, nodes))
	B_Matrix	=	np.zeros((1,nodes))

	A_Matrix[0,0] ,	A_Matrix[-1,-1]	=	1, 1

	Nvp 	=	Grid[:,step-1,-1]		# last time step

	for i in range(0,nodes):

		if (i !=0) and (i != nodes-1):

			D_C_1, D_C_2	=	Diffusion_Coe(DiffusionCoefficient, Nvp[i-1], Nvp[i], Nvp[i+1])

			aE, aW	=	2*D_C_1/delta_x, 2*D_C_2/delta_x
			aP		=	aE + aW + delta_x/delta_t*(1 + kev(Nvp[i]))

			A_Matrix[i,i-1]	=	-aE
			A_Matrix[i,i+1]	=	-aW
			A_Matrix[i,i]	=	aP

		b		=	delta_x/delta_t*(kev(Nvp[i])*NTem(Nvp[i])+dNvsn(Nvp[i])+Nvp[i])
		B_Matrix[0,i]	=	b


	A_Matrix, B_Matrix = np.array(A_Matrix), np.array(B_Matrix)

	return A_Matrix, B_Matrix

def Nv_Sn(Nvp):
	
	if Nvp <= NvL[0]:
		Nv_Np	=	Nvp

	elif	Nvp >= NvL[0] and Nvp <= NvR[0]:
		Nv_Np	=	Nvp

	elif	Nvp >= NvR[0] and Nvp <= NvL[1]:
		Nv_Np	=	Nvp		-	(NvR[0] - NvL[0])

	elif	Nvp >= NvL[1] and Nvp <= NvR[1]:
		Nv_Np	=	NvL[1]	-	(NvR[0] - NvL[0])

	elif	Nvp >= NvR[1] and Nvp <= NvL[2]:
		Nv_Np	=	Nvp		-	(NvR[1] - NvL[1])	-	(NvR[0] - NvL[0])

	elif	Nvp >= NvL[2] and Nvp <= NvR[2]:
		Nv_Np	=	NvL[2]	-	(NvR[1] - NvL[1])	-	(NvR[0] - NvL[0])

	elif	Nvp >= NvR[2]:
		Nv_Np	=	Nvp		-	(NvR[2] - NvL[2])	-	(NvR[1] - NvL[1])	-	(NvR[0] - NvL[0])

	return	Nv_Np

def kev(Nvp):

	K, delta_ = np.zeros(3), np.zeros(3)

	for i in range(3):

		if Nvp > Nv_Sn(NvL[i]):
			K[i] 		=	NvR[i] - NvL[i]
			delta_[i]	=	delta_d[i]

	k	=	np.sum(K)/(np.maximum( Nvp - NTem(Nvp), np.sum(delta_)))
	
	if math.isnan(k):
		k = 0

	return k

def NTem(Nvp):

	if Nv_Sn(NvL[0]) <= Nvp and Nv_Sn(NvL[1]) >= Nvp:
		n 	=	Nv_Sn(NvL[0])

	elif Nv_Sn(NvL[1]) <= Nvp and Nv_Sn(NvL[2]) >= Nvp:
		n 	=	Nv_Sn(NvL[1]) - delta_d[0]

	elif Nv_Sn(NvL[2]) <= Nvp :
		n 	=	Nv_Sn(NvL[2]) - delta_d[0] - delta_d[1]
	
	else:
		n	=	0

	return n

def dNvsn(Nvp):
	
	dNv, dn = np.zeros(3), np.zeros(3)

	for i in range(3):
		if Nvp >= Nv_Sn(NvR[i]):
			dNv[i]	=	NvR[i]	-	NvL[i]	-	delta_d[i]
		elif Nvp < Nv_Sn(NvL[i]):
			dNv[i]	=	0

	for i in range(3):
		dn[i] = dNv[i]*np.minimum(np.maximum(0, (Nvp - Nv_Sn(NvL[i]))/delta_d[i]),1)

	return np.sum(dn)

# OutPut
#=============================================

def SaveOutput(filename, FinalGrid):
	print("Output {}".format(filename))
	np.save(filename, FinalGrid)

# Post - Processing
#=============================================

def findBoundary(DensityList):

	Boundary	=	np.zeros(3)

	for num, Density in enumerate(DensityList):

		if (Density >=	Nv_Sn(NvL[0])) and (Boundary[0] == 0):
			Boundary[0]	=	num
	
		elif (Density >=	Nv_Sn(NvL[1])) and (Boundary[1] == 0):
			Boundary[1]	=	num
	
		elif (Density >=	Nv_Sn(NvL[2])) and (Boundary[2] == 0):
			Boundary[2]	=	num
			break

	return	Boundary

def PlotBoundary(DensityFileName):

	Data	=	np.load(DensityFileName)
	Boundary=	np.zeros((iteration,3))
	for timestep in range(iteration):
		for numberOfBoundary in range(3):
			Boundary[timestep] =	findBoundary(Data[:,timestep,-1])


	Boundary =	(Boundary - nodes/2)/nodes*L
	plt.plot(range(iteration), Boundary[:,0], color = "r")
	plt.plot(range(iteration), Boundary[:,1], color = "g")
	plt.plot(range(iteration), Boundary[:,2], color = "b")

	plt.xlim([0,iteration])
	plt.ylim([-L/2,L/2])
	plt.xlabel("iteration")
	plt.ylabel("location")
	plt.savefig("Boundary.png")

# Run script
#=============================================

def runDiffusion():

	filename	=	"./test"
	TemData		=	np.load("1.5-4.5.npy")
	initialGrid	=	initialModel()
	FinalGrid	=	RuntIteration(initialGrid, TemData[0])

	SaveOutput(filename, FinalGrid)

def runPostProcessing():

	filename	=	"./test.npy"
	# DensityFile		=	np.load(filename)
	# Boundary	=	findBondary(file)
	# Plotboundary(Boundary)
	# print(DensityFile[:50,-1,-1])
	# print(DensityFile[-50:,-1,-1])
	# print(Boundary[0,:])
	PlotBoundary(filename)

def main():
	runDiffusion()
	runPostProcessing()

if __name__ == '__main__':
	main()