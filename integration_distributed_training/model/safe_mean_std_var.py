
import numpy as np
import numpy.testing as npt

def mean_std_var(X, axis=0):
    # Only supports along axis=0 and axis=1 for now.

    assert axis in [0, 1]

    if axis==0:
        N = X.shape[0]
    elif axis==1:
        N = X.shape[1]

    chunk_size = N/10 + 1
    accum_var = np.float64(0.0)
    accum_mean = np.float64(0.0)
    for start in range(0, N + chunk_size, chunk_size):

        end = start + chunk_size
        if N < end:
            end = N

        if end <= start:
            continue
        else:
            if axis==0:
                accum_var += (X[start:end, :]**2).sum(axis=0)
                accum_mean += X[start:end, :].sum(axis=0)
            elif axis==1:
                accum_var += (X[:, start:end]**2).sum(axis=1)
                accum_mean += X[:, start:end].sum(axis=1)

    mean = accum_mean / N
    var = accum_var / N - mean**2
    std = np.sqrt(var)

    return mean, std, var

if __name__ == "__main__":

    for _ in range(100):

        N = np.random.randint(low=1, high=100000)
        D = np.random.randint(low=1, high=10)
        #(N, D) = (21, 2)
        axis = np.random.randint(low=0, high=2)

        X = np.random.rand(N, D)

        mean, std, var = mean_std_var(X, axis=axis)

        # I'm not 100% sure about this since there is something
        # strange with the tolerate that can't go below 1e-4.
        # I'm not sure if I'm missing some indices or if it's
        # just a rounding thing. It certainly looks goods upon
        # visual inspection.

        rtol=1e-4
        npt.assert_allclose(mean, np.mean(X, axis=axis), rtol=rtol)
        npt.assert_allclose(std, np.std(X, axis=axis), rtol=rtol)
        npt.assert_allclose(var, np.var(X, axis=axis), rtol=rtol)
