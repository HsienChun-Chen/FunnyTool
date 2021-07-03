import numpy as np
from matplotlib import pyplot as plt
import os
from timeit import default_timer as timer


def find_nearest(translation, point, cell_box, minDist):
	
	# set cell box and find near cell box 

	x, y, z =	int(point[0]//minDist), int(point[1]//minDist), int(point[2]//minDist)
	Max_x, Max_y, Max_z,	=	len(cell_box),len(cell_box[0]),len(cell_box[0][0])

	AtomList, check_cell	=	[], []

	for near_x in range(3):
		if (x + near_x - 1 >= 0) and (x + near_x -1 < Max_x):
			for near_y in range(3):
				if (y + near_y - 1 >= 0) and (y + near_y -1 < Max_y):
					for near_z in range(3):
						if (z + near_z - 1 >= 0) and (z + near_z -1 < Max_z):
							if type(cell_box[x + near_x - 1, y + near_y - 1, z + near_z - 1]) is not int:
								check_cell = [*check_cell , *cell_box[x + near_x - 1, y + near_y - 1, z + near_z - 1]]


	check_cell	=	np.reshape(check_cell,-1)
	check_cell	=	list(set(check_cell))

	if len(check_cell) <= 1:
		array	=	translation[translation != 0].reshape(-1,3)

	else:
		for i in range(len(check_cell)):
			AtomList.append(translation[check_cell[i]])	
		array	=	np.array(AtomList)


	# find minima distance in near cell box

	array		=	np.asarray(array)
	point		=	np.asarray(point)
	minus		=	np.abs(array - point)
	dist		=	np.zeros(len(minus))

	for n_m, particle in enumerate(minus):

		dist[n_m]	=	np.sqrt(particle[0]**2+particle[1]**2+particle[2]**2)
	distance	=	dist.min()

	return distance

def PointLocation(boxSize, GenerateSteps, numParticle, minDist):

	# generate random PointLocation

	x_lo, y_lo, z_lo	=	2.5,	2.5,	2.5
	x_hi, y_hi, z_hi	=	boxSize[0]-2.5, boxSize[1]-2.5, boxSize[2]-2.5

	step		=	0
	p			=	1
	Output		=	np.zeros((numParticle,3))
	NPoint 		=	len(Output[Output != 0])

	cell_len	=	[]
	for i in range(3):
		cell_len.append(int((boxSize[i]//minDist+1)))

	cell_box	=	np.zeros((cell_len[0], cell_len[1], cell_len[2]), dtype = object)


	while (step < GenerateSteps) and (p < numParticle):

		if step % 1000 == 0:
			print("iteration: {}; particle: {}%".format(step,'%.2f' %(p/numParticle*100)))

		# create a new random point
		newPoint	=	np.zeros(3)
		newPoint[0] = 	"%.3f" %(np.random.randint(x_lo, x_hi) + np.random.rand())
		newPoint[1]	= 	"%.3f" %(np.random.randint(y_lo, y_hi) + np.random.rand())
		newPoint[2]	= 	"%.3f" %(np.random.randint(z_lo, z_hi) + np.random.rand())

		x, y, z 	=	int(newPoint[0]//minDist), int(newPoint[1]//minDist), int(newPoint[2]//minDist)

		if step == 0:
			Output[0]	=	newPoint
			cell_box[x,y,z] = np.append(cell_box[x,y,z],0)	

			step	=	step+1
		# compare new point with other point
		else:
			dist	=	find_nearest(Output, newPoint, cell_box,minDist)

			if dist >= minDist:
				Output[p]	=	newPoint
				cell_box[x,y,z] = np.append(cell_box[x,y,z],p)	
				p	=	p+1

			step	=	step+1

	return Output

def Rotation(numParticle):

	# generate random rotation angle

	r = 0
	Output	=	np.zeros((numParticle,3))

	for r in range(numParticle):
		
		for d in range(3):
			Output[r,d]	=	"%.3f" %((np.random.randint(0,1000)/np.pi)%np.pi)

	return Output

def SeedPosition(boxSize, NumberOfAmole,NumberOfBmole, GenerateSteps, minDist):
	# define pmolecule center location and rotation

	numParticle		=	NumberOfAmole	+	NumberOfBmole
	Output_1 = PointLocation(boxSize, GenerateSteps, numParticle, minDist)
	Output_2 = Rotation(numParticle)

	return Output_1, Output_2
	
class Mole():

	# skip_header & max_rows need to check in mole file
	def __init__(self, Mtype):
		self.type	=	Mtype

	def Atoms(self, PointLocation, Rotation):
		# rotation matrix

		alpha, beta, gamma = Rotation[0], Rotation[1], Rotation[2]
		x, y, z = PointLocation[0], PointLocation[1], PointLocation[2]

		R_z	=	np.asarray([[np.cos(alpha),-np.sin(alpha),0],[np.sin(alpha),np.cos(alpha),0],[0,0,1]])
		R_y	=	np.asarray([[np.cos(beta),0,np.sin(beta)],[0,1,0],[-np.sin(beta),0,np.cos(beta)]])
		R_x	=	np.asarray([[1,0,0],[0,np.cos(gamma),-np.sin(gamma)],[0,np.sin(gamma),np.cos(gamma)]])

		R	=	np.matmul(np.matmul(R_z,R_y),R_x)

		# atom: 1 -O, 2 P, 3 =O, 4 -O, 5 -O, 6 H, 7 H, 8 H

		vector = []
		newVector = []
		pos = []

		if self.type == "H3PO4":
			# mol file 
			molfile = 	"h3po4.mol"
			Data	= 	np.genfromtxt(molfile,	skip_header = 9, max_rows = 8)
			Coords	= 	Data[:,1:4]
			Types	=	np.genfromtxt(molfile,	skip_header =20, max_rows = 8)
			Charges	=	np.genfromtxt(molfile,	skip_header =31, max_rows = 8)
			Charges = 	Charges[:,1]

			atoms = np.zeros((8,10))

			for particle in Coords:
				vector.append(np.asarray(particle) - np.asarray(Coords[1,:]))

		elif self.type == "H2O":
			# mol file 
			molfile = 	"H2O.mol"
			Data	= 	np.genfromtxt(molfile,	skip_header = 8, max_rows = 3)
			Coords	= 	Data[:,1:4]
			Types	=	np.genfromtxt(molfile,	skip_header =14, max_rows = 3)
			Types[:,1]	=	Types[:,1] + 4
			Charges	=	np.genfromtxt(molfile,	skip_header =20, max_rows = 3)
			Charges = 	Charges[:,1]

			atoms = np.zeros((3,10))

			for particle in Coords:
				vector.append(np.asarray(particle) - np.asarray(Coords[0,:]))

		for v in vector:
			newVector.append(np.matmul(R,v))

		newVector	=	np.array(newVector)

		for NewV in newVector:

			pos.append([x+NewV[0],y+NewV[1],z+NewV[2]])


		pos = np.array(pos)
		atoms[:,0] = Types[:,0]
		atoms[:,1] = 1
		atoms[:,2] = Types[:,1]
		atoms[:,3] = Charges[:]
		atoms[:,4:7] = pos[:,0:3]

		return atoms

	def Bonds(self):
		if self.type =="H3PO4":
			bondinfo = np.genfromtxt("h3po4.mol",skip_header = 42, max_rows = 7)

		elif self.type =="H2O":
			bondinfo = np.genfromtxt("H2O.mol",skip_header = 27, max_rows = 2)

		return bondinfo

	def Angles(self):
		if self.type =="H3PO4":
			angleInfo = np.genfromtxt("h3po4.mol",skip_header = 52, max_rows = 9)

		elif self.type =="H2O":
			angleInfo = np.genfromtxt("H2O.mol",skip_header = 33, max_rows = 1)

		return angleInfo

	def Dihedrals(self):
		if self.type =="H3PO4":
			dihedralsInfo = np.genfromtxt("h3po4.mol",skip_header = 64, max_rows = 9)

		else:
			dihedralsInfo = None

		return dihedralsInfo

def List_4(numH3PO4, numH2O, Seed):
	# Seed
	PointLocation, Rotation = Seed

	# AtomList
	AtominH3PO4, AtominH2O	=	8,	3
	numAtoms	=	numH3PO4*AtominH3PO4	+	numH2O*AtominH2O
	AtomList	=	np.zeros((numAtoms, 10))

	for i in range(numH3PO4):
		atoms	=	Mole("H3PO4").Atoms(PointLocation[i], Rotation[i])
		AtomList[AtominH3PO4*i:AtominH3PO4*(i+1),:]	=	atoms
		AtomList[AtominH3PO4*i:AtominH3PO4*(i+1),0]	=	AtomList[AtominH3PO4*i:AtominH3PO4*(i+1),0] + AtominH3PO4*i
		AtomList[AtominH3PO4*i:AtominH3PO4*(i+1),1] =	AtomList[AtominH3PO4*i:AtominH3PO4*(i+1),1] + i

	for j in range(numH2O):
		atoms	=	Mole("H2O").Atoms(PointLocation[numH3PO4+j], Rotation[numH3PO4+j])
		AtomList[AtominH3PO4*numH3PO4+AtominH2O*j:AtominH3PO4*numH3PO4+AtominH2O*(j+1),:]	=	atoms
		AtomList[AtominH3PO4*numH3PO4+AtominH2O*j:AtominH3PO4*numH3PO4+AtominH2O*(j+1),0]	=	AtomList[AtominH3PO4*numH3PO4+AtominH2O*j:AtominH3PO4*numH3PO4+AtominH2O*(j+1),0] + AtominH3PO4*numH3PO4 + AtominH2O*j
		AtomList[AtominH3PO4*numH3PO4+AtominH2O*j:AtominH3PO4*numH3PO4+AtominH2O*(j+1),1]	=	AtomList[AtominH3PO4*numH3PO4+AtominH2O*j:AtominH3PO4*numH3PO4+AtominH2O*(j+1),1] + numH3PO4 + j

	# BondList
	BondinH3PO4, BondinH2O	=	7,	2
	numBonds	=	numH3PO4*BondinH3PO4	+	numH2O*BondinH2O
	BondList	=	np.zeros((numBonds,	4))

	for i in range(numH3PO4):
		bonds	=	Mole("H3PO4").Bonds()
		BondList[BondinH3PO4*i:BondinH3PO4*(i+1), :]	=	bonds
		BondList[BondinH3PO4*i:BondinH3PO4*(i+1), 0]	=	BondList[BondinH3PO4*i:BondinH3PO4*(i+1), 0]	+	BondinH3PO4*i
		BondList[BondinH3PO4*i:BondinH3PO4*(i+1), 2:4]	=	BondList[BondinH3PO4*i:BondinH3PO4*(i+1), 2:4]	+	AtominH3PO4*i

	for j in range(numH2O):
		bonds	=	Mole("H2O").Bonds()
		BondList[BondinH3PO4*numH3PO4+BondinH2O*j:BondinH3PO4*numH3PO4+BondinH2O*(j+1),:]		=	bonds
		BondList[BondinH3PO4*numH3PO4+BondinH2O*j:BondinH3PO4*numH3PO4+BondinH2O*(j+1), 0]		=	BondList[BondinH3PO4*numH3PO4+BondinH2O*j:BondinH3PO4*numH3PO4+BondinH2O*(j+1), 0]	+	BondinH3PO4*numH3PO4 + BondinH2O*j
		BondList[BondinH3PO4*numH3PO4+BondinH2O*j:BondinH3PO4*numH3PO4+BondinH2O*(j+1), 1]		=	BondList[BondinH3PO4*numH3PO4+BondinH2O*j:BondinH3PO4*numH3PO4+BondinH2O*(j+1), 1]	+	3
		BondList[BondinH3PO4*numH3PO4+BondinH2O*j:BondinH3PO4*numH3PO4+BondinH2O*(j+1), 2:4]	=	BondList[BondinH3PO4*numH3PO4+BondinH2O*j:BondinH3PO4*numH3PO4+BondinH2O*(j+1), 2:4]+	AtominH3PO4*numH3PO4 + AtominH2O*j

	# AngleList
	AngleinH3PO4, AngleinH2O	=	9,	1
	numAngles	=	numH3PO4*AngleinH3PO4	+	numH2O*AngleinH2O
	AngleList	=	np.zeros((numAngles, 5))

	for i in range(numH3PO4):
		angles	=	Mole("H3PO4").Angles()
		AngleList[AngleinH3PO4*i:AngleinH3PO4*(i+1), :]	=	angles
		AngleList[AngleinH3PO4*i:AngleinH3PO4*(i+1), 0]	=	AngleList[AngleinH3PO4*i:AngleinH3PO4*(i+1), 0]	+	AngleinH3PO4*i
		AngleList[AngleinH3PO4*i:AngleinH3PO4*(i+1), 2:5]	=	AngleList[AngleinH3PO4*i:AngleinH3PO4*(i+1), 2:5]	+	AtominH3PO4*i	

	for j in range(numH2O):
		amgles	=	Mole("H2O").Angles()
		AngleList[AngleinH3PO4*numH3PO4+j:AngleinH3PO4*numH3PO4+(j+1),:]		=	amgles
		AngleList[AngleinH3PO4*numH3PO4+j:AngleinH3PO4*numH3PO4+(j+1), 0]		=	AngleList[AngleinH3PO4*numH3PO4+j:AngleinH3PO4*numH3PO4+(j+1), 0]	+	AngleinH3PO4*numH3PO4 + j
		AngleList[AngleinH3PO4*numH3PO4+j:AngleinH3PO4*numH3PO4+(j+1), 1]		=	AngleList[AngleinH3PO4*numH3PO4+j:AngleinH3PO4*numH3PO4+(j+1), 1]	+	3
		AngleList[AngleinH3PO4*numH3PO4+j:AngleinH3PO4*numH3PO4+(j+1), 2:5]		=	AngleList[AngleinH3PO4*numH3PO4+j:AngleinH3PO4*numH3PO4+(j+1), 2:5]	+	AtominH3PO4*numH3PO4 + AtominH2O*j

	# Dihedrals
	DuhedralinH3PO4, DuhedralinH2O	=	9,	0
	numDihedrals=	numH3PO4*DuhedralinH3PO4
	DihedralList=	np.zeros((numDihedrals,6))

	for i in range(numH3PO4):
		dihedral = Mole("H3PO4").Dihedrals()
		DihedralList[DuhedralinH3PO4*i:DuhedralinH3PO4*(i+1), :]	=	dihedral
		DihedralList[DuhedralinH3PO4*i:DuhedralinH3PO4*(i+1), 0]	=	DihedralList[DuhedralinH3PO4*i:DuhedralinH3PO4*(i+1), 0]	+	DuhedralinH3PO4*i
		DihedralList[DuhedralinH3PO4*i:DuhedralinH3PO4*(i+1), 2:6]	=	DihedralList[DuhedralinH3PO4*i:DuhedralinH3PO4*(i+1), 2:6]	+	AtominH3PO4*i

	return AtomList, BondList, AngleList, DihedralList

def output(List_4, boxSize, filename):

	print("Start to write data.")

	header	=	("LAMMPS data file via python script, version 1, fuck COVID-19\n\n")
	
	Atoms, Bonds, Angles, Dihedrals = List_4
	
	N		=	[len(Atoms), len(Bonds), len(Angles), len(Dihedrals)]
	Ntypes	=	[Atoms[:,2].max(), Bonds[:,1].max(), Angles[:,1].max(), Dihedrals[:,1].max()]
	
	dataInfo=	("{} atoms\n{} atom types\n{} bonds\n{} bond types\n{} angles\n{} angle types\n{} dihedrals\n{} dihedral types\n\n".format
				(int(N[0]),int(Ntypes[0]),int(N[1]),int(Ntypes[1]),int(N[2]),int(Ntypes[2]),int(N[3]),int(Ntypes[3])))
	boxInfo	=	("0 {} xlo xhi\n0 {} ylo yhi\n0 {} zlo zhi\n\n".format(boxSize[0],boxSize[1],boxSize[2]))

	atomInfo	=	("Atoms # full\n\n")
	bondInfo	=	("\nBonds\n\n")
	AngleInfo	=	("\nAngles\n\n")
	dihedralInfo=	("\nDihedrals\n\n")

	atomfmt		=	["%d","%d","%d","%.5f","%.5f","%.5f","%.5f","%d","%d","%d"]

	if os.path.isfile(filename):
		os.remove(filename)

	with open(filename, "a") as f:
		f.write(header)
		f.write(dataInfo)
		f.write(boxInfo)
		f.write(atomInfo)
		np.savetxt(f, Atoms, fmt = atomfmt)
		f.write(bondInfo)
		np.savetxt(f, Bonds, fmt = "%d")
		f.write(AngleInfo)
		np.savetxt(f, Angles, fmt= "%d")
		f.write(dihedralInfo)
		np.savetxt(f, Dihedrals, fmt = "%d")
		f.close()
		print("Finish !!!!!!")

def main():
	start				=	timer()


	# parameter
	#================================================================
	boxSize				=	[413, 49, 300]
	# xhi, yhi, zhi
	# xlo, ylo, zlo is 0,0,0

	# numH3PO4, numH2O	=	13932, 12348
	numH3PO4, numH2O	=	1393, 1234
	# number of mole

	GenerateSteps		=	1e6
	# max iteration

	outputfileName		=	"test1.data"
	# data name

	minDist		=	5
	# minima distance between molecule center

	# functions
	#================================================================
	Seed	=	SeedPosition(boxSize, numH3PO4,numH2O, GenerateSteps, minDist)
	List	=	List_4(numH3PO4,numH2O,Seed)
	output(List, boxSize, outputfileName)

	end = timer()
	TIME = "%.3f" %(end - start)
	print("generate {} waste {} seconds".format(outputfileName, TIME))

if __name__ == '__main__':
	main()
