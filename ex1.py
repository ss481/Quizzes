import random
import numpy as np

X = random.sample(range(1,1000), 100)

Z = []
for i in range(0,len(X)):
	Z.append(X[i]-np.mean(X))

norm = 10/np.std(Z)

Q = []
for i in range(0,len(X)):
	Q.append(Z[i]*norm)

F = []
for i in range(1,len(X)):
	F.append(Q[i]+1000)

print(F)
print('Mean F: ' + str(np.mean(F)))
print('Standard Deviation F: ' + str(np.std(F)))	

