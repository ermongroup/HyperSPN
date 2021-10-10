import numpy as np

TOY_DATA_PATH = 'toy/toy/toy.%s.data'

def gen_data():
    N=256
    M=10000

    data = np.zeros([M, N])

    for m in range(M):
        d = 1
        p = np.zeros(N) + 0.5
        while d < N:
            d *= 2
            p2 = np.zeros(N)
            p2[::2] = p[:N//2]
            p2[1::2] = p[:N//2]
            p = p2 + np.random.uniform(low=-0.2, high=0.2, size=(N))
        data[m,:] = p

    data = np.random.rand(M, N) > data
    data = data.astype(int)
    return data

if __name__ == '__main__':
    np.random.seed(0)
    data = gen_data()
    np.random.shuffle(data)

    train_sz, valid_sz = (int)(0.7*len(data)), (int)(0.1*len(data))
    train_data, valid_data, test_data = data[:train_sz], data[train_sz:train_sz+valid_sz], data[train_sz+valid_sz:]
    for mode, dat in [('train', train_data), ('valid', valid_data), ('test', test_data)]:
        filename = TOY_DATA_PATH % mode
        np.savetxt(filename, dat, fmt="%i", delimiter=",")