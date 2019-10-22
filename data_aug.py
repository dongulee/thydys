import numpy as np

dir_ = 'data/pad_aug3/'
X = np.fromfile(dir_ + 'X4.dat', dtype=float)
Y = np.fromfile(dir_ + 'Y4.dat', dtype=float)
X = X.reshape([-1, 7680])
Y = Y.reshape([-1, 4])

# Random shuffle
s = np.arange(len(X))
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
    if t['y'][3] >= 1.8:
        cnt1 += 1
        tuples_1.append(t)
    else:
        cnt0 += 1
        tuples_0.append(t)


if cnt1 > cnt0: 
    # augment lacking class to dataset
    diff = cnt1-cnt0
    print("diff{}".format(diff))
    for i in range(0,cnt0):
        idx = i % cnt0
        tuples.append(tuples_0[idx])
    for i in range(cnt0, diff):
        idx = np.random.randint(211) % cnt0
        tuples.append(tuples_0[idx])

elif cnt1 < cnt0:
    # augment lacking class to dataset
    diff = cnt0-cnt1
    print("diff{}".format(diff))
    for i in range(0,cnt1):
        idx = i % cnt1
        tuples.append(tuples_1[idx])
    for i in range(cnt1, diff):
        idx = np.random.randint(211) % cnt1
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

X.tofile(dir_+'X4_aug.dat')
Y.tofile(dir_+'Y4_aug.dat')

