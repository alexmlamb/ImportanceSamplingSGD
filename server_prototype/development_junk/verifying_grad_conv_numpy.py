
import numpy as np


def conv(inputs, filters):

    assert len(inputs.shape) == 4, "inputs.shape is %s" % str(inputs.shape)
    assert len(filters.shape) == 4
    (N, C, H, W) = inputs.shape
    (F, _, h, w) = filters.shape
    assert (F, C, h, w) == filters.shape

    result = np.zeros((N, F, H-h+1, W-w+1))

    for n in range(N):
        for c in range(C):
            for f in range(F):
                for i in range(H-h+1):
                    for j in range(W-w+1):
                        result[n, f, i, j] = (inputs[n, c, i:i+h, j:j+w] * filters[f, c, :, :]).sum()

    return result



def manual_extended_bp_filters(inputs, filters_shape, bp_output):

    # yeah, the filters aren't necessary here, but let's keep them
    # because they give the shape and it's easier than writing incorrect code

    assert len(inputs.shape) == 4, "inputs.shape is %s" % str(inputs.shape)
    assert len(filters_shape) == 4
    (N, C, H, W) = inputs.shape
    (F, _, h, w) = filters_shape
    assert (F, C, h, w) == filters_shape
    assert (N, F, H-h+1, W-w+1) == bp_output.shape

    extended_bp_filters = np.zeros((N, F, C, h, w))

    for n in range(N):
        for c in range(C):
            for f in range(F):
                extended_bp_filters[n, f, c, :, :] = conv(inputs[n, c, :, :].reshape((1,1,H,W)), bp_output[n, f, :, :].reshape((1,1,H-h+1,W-w+1)))

    bp_filters = extended_bp_filters.sum(axis=0)
    return (bp_filters, extended_bp_filters)


def bp_filters_squared_norm_SOUND_00(inputs, filters_shape, bp_output):

    (_, extended_bp_filters) = manual_extended_bp_filters(inputs, filters_shape, bp_output)
    return np.array([(extended_bp_filters[n, :, :, :, :]**2).sum() for n in range(extended_bp_filters.shape[0])])

def bp_filters_squared_norm_EXPERIMENTAL_01(inputs, filters_shape, bp_output):

    E = conv( (inputs**2).sum(axis=1, keepdims=True), (bp_output**2).sum(axis=1, keepdims=True) )
    print "E.shape : %s" % str(E.shape)
    return E.reshape((E.shape[0], -1)).sum(axis=1)


def bp_filters_squared_norm_EXPERIMENTAL_02(inputs, filters_shape, bp_output):

    assert len(inputs.shape) == 4, "inputs.shape is %s" % str(inputs.shape)
    assert len(filters_shape) == 4
    (N, C, H, W) = inputs.shape
    (F, _, h, w) = filters_shape
    assert (F, C, h, w) == filters_shape
    assert (N, F, H-h+1, W-w+1) == bp_output.shape

    E = np.zeros((N, 1, h, w))

    for n in range(N):

        A = (inputs**2)[n, np.newaxis, :, :, :].sum(axis=1, keepdims=True)
        B = (bp_output**2)[n, np.newaxis, :, :, :].sum(axis=1, keepdims=True)

        E[n, 0, :, :] = conv(A, B)
    

    return E.reshape((E.shape[0], -1)).sum(axis=1)



def run_experiment():

    (N, F, C) = (32, 10, 6)
    (H, W)    = (16, 12)
    (h, w)    = (4, 2)
    filters_shape = (F, C, h, w)

    want_all_random_data = True
    if want_all_random_data:
        inputs = 0.1*np.random.randn(N, C, H, W)
        bp_output = 0.1*np.random.randn(N, F, H-h+1, W-w+1)
    else:
        inputs = np.ones((N, C, H, W))
        bp_output = np.zeros((N, F, H-h+1, W-w+1))

        def prod(L):
            reduce(lambda x, y:x*y, L)

        (i,j) = (0,0)
        #i = np.random.randint(low=0, high=N*C*H*W)
        j = np.random.randint(low=0, high=N*F*(H-h+1)*(W-w+1))
        inputs.reshape((-1))[i] = 1.0
        bp_output.reshape((-1))[j] = 1.0


    R0 = bp_filters_squared_norm_SOUND_00(inputs, filters_shape, bp_output)
    R1 = bp_filters_squared_norm_EXPERIMENTAL_01(inputs, filters_shape, bp_output)
    R2 = bp_filters_squared_norm_EXPERIMENTAL_02(inputs, filters_shape, bp_output)

    print "bp_filters_squared_norm_SOUND_00 :"
    print R0
    print ""
    print "bp_filters_squared_norm_EXPERIMENTAL_01 :"
    print R1
    print ""
    print "bp_filters_squared_norm_EXPERIMENTAL_02 :"
    print R2
    print ""
    print "R0 / R1 :"
    print R0 / R1
    print "R0 / R2 :"
    print R0 / R2


if __name__ == "__main__":
    run_experiment()
