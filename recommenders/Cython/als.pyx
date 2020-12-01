#############################################################################################################################
# taken from https://github.com/benfred/implicit/blob/d54e3617ce54a7fa1759d156a8c389ffd55d6c45/implicit/_als.pyx#L125
#############################################################################################################################
import numpy as np
import cython
from cython cimport floating
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

# requires scipy v0.16
cimport scipy.linalg.cython_lapack as cython_lapack
cimport scipy.linalg.cython_blas as cython_blas

# lapack/blas wrappers for cython fused types
cdef inline void axpy(int * n, floating * da, floating * dx, int * incx, floating * dy,
                      int * incy) nogil:
    if floating is double:
        cython_blas.daxpy(n, da, dx, incx, dy, incy)
    else:
        cython_blas.saxpy(n, da, dx, incx, dy, incy)

cdef inline void symv(char *uplo, int *n, floating *alpha, floating *a, int *lda, floating *x,
                      int *incx, floating *beta, floating *y, int *incy) nogil:
    if floating is double:
        cython_blas.dsymv(uplo, n, alpha, a, lda, x, incx, beta, y, incy)
    else:
        cython_blas.ssymv(uplo, n, alpha, a, lda, x, incx, beta, y, incy)

cdef inline floating dot(int *n, floating *sx, int *incx, floating *sy, int *incy) nogil:
    if floating is double:
        return cython_blas.ddot(n, sx, incx, sy, incy)
    else:
        return cython_blas.sdot(n, sx, incx, sy, incy)

cdef inline void scal(int *n, floating *sa, floating *sx, int *incx) nogil:
    if floating is double:
        cython_blas.dscal(n, sa, sx, incx)
    else:
        cython_blas.sscal(n, sa, sx, incx)

cdef inline void posv(char * u, int * n, int * nrhs, floating * a, int * lda, floating * b,
                      int * ldb, int * info) nogil:
    if floating is double:
        cython_lapack.dposv(u, n, nrhs, a, lda, b, ldb, info)
    else:
        cython_lapack.sposv(u, n, nrhs, a, lda, b, ldb, info)

cdef inline void gesv(int * n, int * nrhs, floating * a, int * lda, int * piv, floating * b,
                      int * ldb, int * info) nogil:
    if floating is double:
        cython_lapack.dgesv(n, nrhs, a, lda, piv, b, ldb, info)
    else:
        cython_lapack.sgesv(n, nrhs, a, lda, piv, b, ldb, info)


@cython.cdivision(True)
@cython.boundscheck(False)
def least_squares_cg(Cui, floating[:, :] X, floating[:, :] Y, float regularization,
                     int num_threads=0, int cg_steps=3):
    dtype = np.float64 if floating is double else np.float32
    cdef int[:] indptr = Cui.indptr, indices = Cui.indices
    cdef double[:] data = Cui.data

    cdef int users = X.shape[0], N = X.shape[1], u, i, index, one = 1, it
    cdef floating confidence, temp, alpha, rsnew, rsold
    cdef floating zero = 0.

    cdef floating[:, :] YtY = np.dot(np.transpose(Y), Y) + regularization * np.eye(N, dtype=dtype)

    cdef floating * x
    cdef floating * p
    cdef floating * r
    cdef floating * Ap

    # allocate temp memory for each thread
    Ap = <floating *> malloc(sizeof(floating) * N)
    p = <floating *> malloc(sizeof(floating) * N)
    r = <floating *> malloc(sizeof(floating) * N)
    try:
        # start from previous iteration
        x = &X[u, 0]

        # calculate residual r = (YtCuPu - (YtCuY.dot(Xu)
        temp = -1.0
        symv("U", &N, &temp, &YtY[0, 0], &N, x, &one, &zero, r, &one)

        for index in range(indptr[u], indptr[u + 1]):
            i = indices[index]
            confidence = data[index]
            temp = confidence - (confidence - 1) * dot(&N, &Y[i, 0], &one, x, &one)
            axpy(&N, &temp, &Y[i, 0], &one, r, &one)

        memcpy(p, r, sizeof(floating) * N)
        rsold = dot(&N, r, &one, r, &one)

        for it in range(cg_steps):
            # calculate Ap = YtCuYp - without actually calculating YtCuY
            temp = 1.0
            symv("U", &N, &temp, &YtY[0, 0], &N, p, &one, &zero, Ap, &one)

            for index in range(indptr[u], indptr[u + 1]):
                i = indices[index]
                confidence = data[index]
                temp = (confidence - 1) * dot(&N, &Y[i, 0], &one, p, &one)
                axpy(&N, &temp, &Y[i, 0], &one, Ap, &one)

            # alpha = rsold / p.dot(Ap);
            alpha = rsold / dot(&N, p, &one, Ap, &one)

            # x += alpha * p
            axpy(&N, &alpha, p, &one, x, &one)

            # r -= alpha * Ap
            temp = alpha * -1
            axpy(&N, &temp, Ap, &one, r, &one)

            rsnew = dot(&N, r, &one, r, &one)
            if rsnew < 1e-10:
                break

            # p = r + (rsnew/rsold) * p
            temp = rsnew / rsold
            scal(&N, &temp, p, &one)
            temp = 1.0
            axpy(&N, &temp, r, &one, p, &one)

            rsold = rsnew
    finally:
        free(p)
        free(r)
        free(Ap)