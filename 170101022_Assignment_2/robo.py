from pomegranate import *
import numpy as np
#  d1 = DiscreteDistribution({'A' : 0.35, 'C' : 0.20, 'G' : 0.05, 'T' : 0.40})
# d2 = DiscreteDistribution({'A' : 0.25, 'C' : 0.25, 'G' : 0.25, 'T' : 0.25})
# >>> d3 = DiscreteDistribution({'A' : 0.10, 'C' : 0.40, 'G' : 0.40, 'T' : 0.10})
# >>>
# >>> s1 = State(d1, name="s1")
# >>> s2 = State(d2, name="s2")
# >>> s3 = State(d3, name="s3")
# >>>
# >>> model = HiddenMarkovModel('example')
# >>> model.add_states([s1, s2, s3])
# >>> model.add_transition(model.start, s1, 0.90)
# >>> model.add_transition(model.start, s2, 0.10)
# >>> model.add_transition(s1, s1, 0.80)
# >>> model.add_transition(s1, s2, 0.20)
# >>> model.add_transition(s2, s2, 0.90)
# >>> model.add_transition(s2, s3, 0.10)
# >>> model.add_transition(s3, s3, 0.70)
# >>> model.add_transition(s3, model.end, 0.30)
# >>> model.bake()
# >>>
# >>> print(model.log_probability(list('ACGACTATTCGAT')))
# -22.73896159971087
# >>> print(", ".join(state.name for i, state in model.viterbi(list('ACGACTATTCGAT'))[1]))



mat = [[0 for col in range(16)] for row in range(4)]


mat[0][4] = 1
mat[0][10] = 1
mat[0][14] = 1
mat[1][0] = 1
mat[1][1] = 1
mat[1][4] = 1
mat[1][6] = 1
mat[1][7] = 1
mat[1][9] = 1
mat[1][11] = 1
mat[1][13] = 1
mat[1][14] = 1
mat[1][15] = 1
mat[2][0] = 1
mat[2][4] = 1
mat[2][6] = 1
mat[2][7] = 1
mat[2][13] = 1
mat[2][14] = 1
mat[3][2] = 1
mat[3][6] = 1
mat[3][11] = 1
mat[0][4] = 1
mat[0][10] = 1
mat[0][14] = 1
mat[1][0] = 1
mat[1][1] = 1
mat[1][4] = 1
mat[1][6] = 1
mat[1][7] = 1
mat[1][9] = 1
mat[1][11] = 1
mat[1][13] = 1
mat[1][14] = 1
mat[1][15] = 1
mat[2][0] = 1
mat[2][4] = 1
mat[2][6] = 1
mat[2][7] = 1
mat[2][13] = 1
mat[2][14] = 1
mat[3][2] = 1
mat[3][6] = 1
mat[3][11] = 1


mapping = dict()
walls = dict()

state = 1
for i in range(4):
	for j in range(16):
		if mat[i][j]==0:
			mapping[(i,j)] = state
			wall = [0,0,0,0]
			if(i==0):
				wall[0] = 1
			if(i==3):
				wall[2] = 1
			if(j==0):
				wall[3] = 1
			if(j==15):
				wall[1] = 1

			if (i>0 and mat[i-1][j] == 1):
				wall[0] = 1
			if (j<15 and mat[i][j+1] == 1):
				wall[1] = 1
			if (i<3 and mat[i+1][j] == 1):
				wall[2] = 1
			if (j>0 and mat[i][j-1] == 1):
				wall[3] = 1

			walls[state] = wall 
			state += 1

neighbours = dict()
for i in range(4):
	for j in range(16):
		if mat[i][j]==0:
			neigh_list = list()

			if (i>0 and mat[i-1][j] == 0):
				neigh_list.append(mapping[(i-1,j)])
			if (j<15 and mat[i][j+1] == 0):
				neigh_list.append(mapping[(i,j+1)])
			if (i<3 and mat[i+1][j] == 0):
				neigh_list.append(mapping[(i+1,j)])
			if (j>0 and mat[i][j-1] == 0):
				neigh_list.append(mapping[(i,j-1)])
			neighbours[mapping[(i,j)]] = neigh_list
		

#print(neighbours)
transitionMat = []
for i in range(1,43):
	row = [0 for j in range(1,43)]
	neigh_list  = neighbours[i]
	if len(neigh_list)==0:
		#print(neigh_list)
		transitionMat.append(row)
		continue
	probabTrans = 1/len(neigh_list)
	for neigh in neigh_list:
		row[neigh-1] = probabTrans

	transitionMat.append(row)

# for i in [0,1]:
# 	for j in [0,1]:
# 		for k in [0,1]:
# 			for l in [0,1]:
error = 0.1
ObserMat = dict()
for N in range(0,2):
	for E in range(0,2):
		for S in range (0,2):
			for W in range(0,2):
				key = str(N) + str(E) + str(S) + str(W); 
				temp = [[0 for col in range(42)] for row in range(42)]

				for i in range(1,43):
					disp = 0
					if N != walls[i][0]:
						disp += 1 
					if E != walls[i][1]:
						disp += 1
					if S != walls[i][2]:
						disp += 1
					if W != walls[i][3]:
						disp += 1
					temp[i-1][i-1] = pow(error,disp) * pow(1-error,4-disp)

				ObserMat[key] = temp


#print(ObserMat['1011'])
transitionMat = np.array(transitionMat)
matTranspose = transitionMat.transpose()
# print(matTranspose.shape)

start = [1/42 for i in range(42)]
start = np.array(start)
# print(start.shape)
 
#filtering
def filter(obsSeq,start,matTranspose,ObserMat,t):
	if t==1:
		prev = start
	else:
		prev = filter(obsSeq,start,matTranspose,ObserMat,t-1)
	
	currObser = obsSeq[t-1]
	oMat = ObserMat[currObser]
	oMat = np.array(oMat)
	predict = np.matmul(matTranspose,prev)
	filtered = np.matmul(oMat,predict)
	sum = 0
	for i in range(42):
		sum += filtered[i]
	for i in range(42):
		filtered[i] = filtered[i]/sum

	return filtered

#matrix to back-track most likeli sequence
backTrack = []

#maxlikelihood sequnce inferencing
def maxlikeliehood(obsSeq,start,transitionMat,ObserMat,t):
	if t==1:
		prev = start
	else:
		prev = maxlikeliehood(obsSeq,start,transitionMat,ObserMat,t-1)

	currMax = [0 for i in range(42)]
	mostPrev = [-1 for i in range(42)]
	for i in range(42):
		for j in range(42):
			if prev[j]*transitionMat[j][i]>currMax[i]:
				currMax[i] = max(currMax[i],prev[j]*transitionMat[j][i])
				mostPrev[i] = j

	backTrack.append(mostPrev)
	currObser = obsSeq[t-1]
	oMat = ObserMat[currObser]
	oMat = np.array(oMat)
	newMat = [0 for i in range(42)]
	for i in range(42):
		newMat[i] = currMax[i]*oMat[i][i]

	sum = 0
	for i in range(42):
		sum += newMat[i]
	for i in range(42):
		newMat[i] = newMat[i]/sum

	return newMat


obsSeq = [] 
  
# number of elemetns as input 
n = int(input("Enter length of observation sequence: ")) 
  
# iterating till the range 
for i in range(0, n): 
    ele = str(input()) 
  
    obsSeq.append(ele) # adding the element 
      
print("observation sequence is: ", obsSeq) 


# obsSeq = ['1011','1010','1000','1100']
# obsSeq1 = ['1011']


prob = filter(obsSeq,start,matTranspose,ObserMat,len(obsSeq))

most_probable = list()
for i,item in enumerate(prob):
	
	if(item == prob.max()):
		most_probable.append(i+1)

print("Mostlikeli locations of robot: ", most_probable)




finalProbab = maxlikeliehood(obsSeq,start,transitionMat,ObserMat,len(obsSeq))
#lastState = np.argmax(finalProbab)
print(finalProbab[lastState])
maxlikeliseq = []

maxlikeliseq.append(lastState+1)
#print(lastState)

for i in reversed(range(1,len(obsSeq))):
	lastState = backTrack[i][lastState]
	maxlikeliseq.append(lastState+1)
	
maxlikeliseq.reverse()
print("most likelisequence is: ",maxlikeliseq)










#print(transitionMat)




# def observation(sample):
# 	D = dict()
# 	for N in range(0,2):
# 		for E in range(0,2):
# 			for S in range (0,2):
# 				for W in range(0,2):
# 					key = str(N) + str(E) + str(S) + str(W); 
# 					disp = 0
					
# 					if N != walls[sample][0]:
# 						disp += 1 
# 					if E != walls[sample][1]:
# 						disp += 1
# 					if S != walls[sample][2]:
# 						disp += 1
# 					if W != walls[sample][3]:
# 						disp += 1
# 					D[key] = pow(error,disp) * pow(1-error,4-disp)
# 	return D

# starts = numpy.array([1/42 for i in range(42)])
# state_list = list()
# for i in range(1,43):
# 	state_list.append(State(DiscreteDistribution(observation(i)), name="s"+str(i)))

# model = HiddenMarkovModel('Localization',state_list[0])
# model.add_states(state_list)


# for i in range(1,43):
# 	for neighbour in neighbours[i]:
# 		total_n = len(neighbours[i])
# 		model.add_transition(state_list[i-1],state_list[neighbour-1],1/total_n)


# model.bake()

# # print(model.predict(('1011','1010'), algorithm='map'))


# print(model.log_probability(['1011','1010']))

# print(", ".join(state.name for i, state in model.predict(('0101', '0001'))[1]))