"""Independent dense Kronecker/projector/SVD oracle, restricted to L <= 8."""
import numpy as np

MATRICES = dict(
    I=np.eye(2, dtype=complex),
    X=np.array([[0, 1], [1, 0]], dtype=complex),
    Y=np.array([[0, -1j], [1j, 0]], dtype=complex),
    Z=np.diag([1, -1]).astype(complex),
)


def matrix(word):
    if len(word) > 8:
        raise ValueError("Dense reference restricted to L <= 8")
    result = np.ones((1, 1), complex)
    for a in reversed(word):
        result = np.kron(result, MATRICES[a])
    return result


def initial_state(L, initial):
    if not 1 <= L <= 8:
        raise ValueError("Dense reference restricted to L <= 8")
    psi = np.zeros(2**L, complex)
    if initial == "zero":
        psi[0] = 1
    elif initial == "plus":
        psi[:] = 2**(-L / 2)
    elif initial == "ghz":
        psi[0] = psi[-1] = 2**(-0.5)
    else:
        raise ValueError(initial)
    return psi


def project(psi, operator, outcome):
    unnormalized = (psi + outcome * (operator @ psi)) / 2
    probability = float(np.vdot(unnormalized, unnormalized).real)
    if probability < 1e-13:
        raise ValueError("Zero Born probability")
    return unnormalized / np.sqrt(probability), probability


def entropy(psi, subset):
    L = (len(psi)).bit_length() - 1
    axes = [L - 1 - k for k in subset]
    other = [i for i in range(L) if i not in axes]
    tensor = psi.reshape([2] * L).transpose(axes + other)
    values = np.linalg.svd(tensor.reshape(2**len(axes), -1),
                           compute_uv=False)**2
    values = values[values > 1e-14]
    return float(-np.sum(values * np.log2(values)))


def pauli_matrix(p, L):
    word = "".join("IXZY"[((p.x >> k) & 1) + 2 * ((p.z >> k) & 1)]
                   for k in range(L))
    canonical = (p.x & p.z).bit_count() % 4
    return (1j)**((p.phase - canonical) % 4) * matrix(word)
