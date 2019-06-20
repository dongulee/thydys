import numpy as np

dir_ = 'data/15D15m/'
X = np.fromfile(dir_ + 'X.dat', dtype=float)
Y = np.fromfile(dir_ + 'Y.dat', dtype=float)
X = X.reshape([96, 2880])
Y = Y.reshape([96, 1])

# Random shuffle
s = np.arange(96)
np.random.shuffle(s)

X = X[s]
Y = Y[s]

# Data distribution normalization
tuples =[]

for i in range(len(X)):
    tuples.append({
        'x':X[i],
        'y':Y[i]
        })
cnt1 = 0
cnt0 = 0
tuples_0 = []
tuples_1 = []
for t in tuples:
    if t['y'][0] >= 1.8:
        cnt1 += 1
        tuples_1.append(t)
    else:
        cnt0 += 1
        tuples_0.append(t)


if cnt1 > cnt0: 
    # augment lacking class to dataset
    diff = cnt1-cnt0
    print("diff{}".format(diff))
    for i in range(0,diff):
        idx = i % cnt0
        tuples.append(tuples_0[idx])

elif cnt1 < cnt0:
    # augment lacking class to dataset
    diff = cnt0-cnt1
    print("diff{}".format(diff))
    for i in range(0,diff):
        idx = i % cnt1
        tuples.append(tuples_1[idx])


print(len(tuples_0))
print(len(tuples_1))
print(len(tuples))
#print([t['y'] for t in tuples])
''' 
elif cnt1 < cnt0:
else:
s = np.arange(99)



# Random shuffle
np.random.shuffle(s)

X = X[s]
Y = Y[s]

X.tofile('X_aug.dat')
Y.tofile('Y_aug.dat')
'''
X = np.array([ t['x'] for t in tuples])
#print(X.shape)
Y = np.array([ t['y'] for t in tuples])

s = np.arange(len(X))
np.random.shuffle(s)
X = X[s]
Y = Y[s]

X.tofile('X_aug.dat')
Y.tofile('Y_aug.dat')

