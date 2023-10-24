import numpy as np
import scipy
import matplotlib.pyplot as plt

### Utility functions ###

def psvd(M, Dmax=None, atol=1e-10):
    """Perform partial SVD on `M`. Cut all singular
    values/vectors exceeding `Dmax` or `atol`.
    Finally, renormalize `S` matrix."""

    U, S, Vd = np.linalg.svd(M, full_matrices=False)

    if Dmax:
        D = S.shape[0]
        if Dmax <= D:
            S = S[:Dmax]
            U = U[:, :Dmax]
            Vd = Vd[:Dmax, :]

    if atol:
        D = S.shape[0]
        Dn = S[ S > atol ].shape[0]
        if Dn <= D:
            S = S[:Dn]
            U = U[:, :Dn]
            Vd = Vd[:Dn, :]

    norm = np.linalg.norm(S)
    err = 1 - norm

    S /= norm # renormalize singular values

    return (U, S, Vd)

def expect(M, op):
    """Calc. expectation value of `op` at site with tensor `M`.
    Note: Implicitly assumed all tensors left (right) to `M` are
    left (right) normalized."""
    trM = np.einsum('ijk,ijm->km', M.conj(), M)
    exp = np.einsum('ji,ji->', op, trM)
    return exp

def expect_all(mps, op):
    """Calc. expectation value of `op` at all
    sites in `mps` (assumed in leftcanonical form)."""
    mps = canonical(mps)
    exps = []

    # first site: only one lambda to right
    M = np.einsum('ijk,jln->ilk', mps[0], mps[1])
    exp = np.real_if_close(expect(M, op))
    exps.append(exp)

    # intermediate sites: mult lambda from left and right
    for i in range(2,len(mps)-2,2):
        M = np.einsum('oin,ijk,jln->olk', mps[i-1], mps[i], mps[i+1])
        exp = np.real_if_close(expect(M, op))
        exps.append(exp)

    # last site: only one lambda to left
    M = np.einsum('ijn,jkl->ikl', mps[-2], mps[-1])
    exp = np.real_if_close(expect(M, op))
    exps.append(exp)

    return exps

def entropy_all(mps):
    """Calc. entanglement entropy of Schmidt decomposition of `mps` at
    all sites as the von Neumann entropy of eigenvalues at each site.
    Note: `mps` is assumed to be leftcanonical."""
    mps, sings = sweep(mps, sing=True)
    return [np.sum([-x**2 * np.log2(x**2) for x in sing]) for sing in sings]

### Matrix Product State ###

class MPS:
    """Matrix product state on L sites.
    Shapes [a,b,c] where a,b are left, right bonds
    and c physical index """

    def __init__(self, tensors):
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors)
    
    def __getitem__(self, idx):
        return self.tensors[idx]

    @property
    def shapes(self):
        return [t.shape for t in self.tensors]

### Matrix Product Operator ###

class MPO:
    """Matrix product operator

    `tensors` is list of site operators in the tensor product.
    Each site tensor has 4 indices: [a(i-1), a(i), oi, oi'], where
    oi is top and oi' is bottom physical index """

    def __init__(self, tensors):
        self.tensors = tensors

    def __mul__(self, other):

        if isinstance(other, MPS):
            """Contract this MPO (top) with `other` MPS (bottom)"""
            tensors = []
            for i in range(len(other)):
                W = self.tensors[i]
                M = other.tensors[i]

                N = np.einsum('ijkl,mnl->imjnk', W, M)
                i,m,j,n,k = N.shape
                N = np.reshape(N, (i*m,j*n,k))
                tensors.append(N)
            return MPS(tensors)

        elif isinstance(other, MPO):
            """Contract this MPO (top) with `other` MPO (bottom)"""
            tensors = []
            for i in range(len(other)):
                W1 = self.tensors[i]
                W2 = other.tensors[i]

                W = np.einsum('ijkl,mnlo->imjnko', W1, W2)
                i,m,j,n,k,o = W.shape
                W = np.reshape(W, (i*m,j*n,k,o))
                tensors.append(W)
            return MPO(tensors)
        else:
            raise Exception("Unknown product.")

    def __len__(self):
        return len(self.tensors)
    
    def __getitem__(self, idx):
        return self.tensors[idx]

### MPS conversions ###

def leftcanonical_from_physical(state, Dmax=None):
    """Create MPS in leftcanonical form from
    one large tensor `state` describing entire system."""
    tensors = []

    L = len(state.shape)
    d = state.shape[0]

    Mm = np.reshape(state, (d,-1))
    U,S,Vd = psvd(Mm, Dmax)

    A = np.transpose(np.reshape(U, (1,d,-1)), (0,2,1))
    tensors.append(A)

    for i in range(1,L):
        l = len(S)
        d = state.shape[i]

        Mm = np.reshape(np.diag(S) @ Vd, (d*l,-1))
        U,S,Vd = psvd(Mm, Dmax)

        A = np.transpose(np.reshape(U, (l,d,-1)), (0,2,1))
        tensors.append(A)

    # Remaining scalar S@Vd might introduce sign. Must absorb.
    tensors[-1] *= S @ Vd

    return MPS(tensors)

def leftcanonical(mps, Dmax=None, sing=False):
    """Convert `mps` assumed in rightcanonical form
    into leftcanonical form. If `sing` is True return
    additionally list of singular values at each site"""
    tensors = []
    sings = []

    M = mps[0]
    l,r,d = mps[0].shape
    L = len(mps)

    Mm = np.reshape(np.transpose(M, (2,0,1)), (d*1, r)) # ((sigmai, ai-1), ai)
    U,S,Vd = psvd(Mm, Dmax=Dmax)

    A = np.transpose(np.reshape(U, (d,1,-1)), (1,2,0)) # -1: floating to right
    tensors.append(A)
    sings.append( S )

    for i in range(1,L):
        M = mps[i]
        Mm = np.einsum('ij,jkl->ikl', np.diag(S) @ Vd, M)
        l,r,d = Mm.shape

        Mm = np.reshape(np.transpose(Mm, (2,0,1)), (d*l,r)) # ((sigmai, ai-1), ai)
        U,S,Vd = psvd(Mm, Dmax=Dmax)
        
        A = np.transpose(np.reshape(U, (d,l,-1)), (1,2,0))
        tensors.append(A)
        sings.append( S )

    # Remaining scalar S@Vd might introduce sign. Must absorb.
    tensors[-1] *= S @ Vd

    if sing:
        return MPS(tensors), sings[:-1] # last singular values have been absorbed above.
    return MPS(tensors)

def rightcanonical(mps, Dmax=None, sing=False):
    """Convert `mps` assumed in leftcanonical form
    into rightcanonical form. If `sing` is True return
    additionally list of singular values at each site"""
    tensors = []
    sings = []

    M = mps[-1]
    l,r,d = mps[-1].shape
    L = len(mps)

    Mm = np.reshape(np.transpose(M, (0,2,1)), (l, d*1)) # (ai-1, (sigmai, 1))
    U,S,Vd = psvd(Mm, Dmax=Dmax)

    B = np.transpose(np.reshape(Vd, (-1,d,1)), (0,2,1)) # -1: floating to left
    tensors.append(B)
    sings.append( S )

    for i in range(L-2,-1,-1):
        M = mps[i]
        Mm = np.einsum('ijk,jl->ilk', M, U @ np.diag(S))
        l,r,d = Mm.shape

        Mm = np.reshape(np.transpose(Mm, (0,2,1)), (l, d*r)) # (ai-1, (sigmai, ai))
        U,S,Vd = psvd(Mm, Dmax=Dmax)
        
        B = np.transpose(np.reshape(Vd, (-1,d,r)), (0,2,1))
        tensors.append(B)
        sings.append( S )

    # Remaining scalar U@S might be negative. Must absorb.
    tensors[-1] *= U @ S

    if sing:
        return MPS(tensors[::-1]), sings[::-1]
    return MPS(tensors[::-1])

def sweep(mps, Dmax=None, sing=False):
    """Convert `mps` assumed in leftcanonical form
    into right->leftcanonical form. If sing is True, additionally
    return singular values of the last conversion (R->L)
    """
    # print(mps[0])
    mps = rightcanonical(mps, Dmax)
    mps, sings = leftcanonical(mps, Dmax, sing=True)
    if sing:
        return mps, sings
    return mps

def canonical(mps, Dmax=None):
    """Convert `mps` assumed in leftcanonical form
    into Vidal notation by using singular values after one sweep"""
    mps, sings = sweep(mps, sing=True)
    tensors = mps.tensors.copy() # make copy of tensor list

    for i in range(1, len(mps.tensors)): # left-mult. all A tensors by inv. lambda matrices
        tensors[i] = np.einsum('ij,jkl->ikl', np.diag(1/sings[i-1]), mps.tensors[i])
    for i in range(len(mps.tensors)-1): # insert lambdas between gammas
        tensors.insert(1+i*2, np.diag(sings[i])[:,:,None])
    return MPS(tensors)

### Time evolution ###

def one_site_from_two_site_tensors(hs, tau, L, d):
    """First order Trotter expansion
    Turn sum of two-qubit-operators `hs` into
    list of single-site tensors `U1`, `U2` into MPO
    """
    tensors = []
    for h in hs:
        O = scipy.linalg.expm(-1j * tau * h) # get matrix exponential
        P = np.reshape(np.transpose(np.reshape(O, (d,d,d,d)), (0,2,1,3)), (d**2, d**2))
        U,S,Vd = psvd(P)

        U1 = np.transpose(np.reshape(U @ np.diag(np.sqrt(S)), (d,d,-1,1)), (3,2,0,1))
        U2 = np.transpose(np.reshape(np.diag(np.sqrt(S)) @ Vd, (-1,d,d,1)), (0,3,1,2))

        tensors.append(U1)
        tensors.append(U2)

    return tensors

iseven = lambda x: x % 2 == 0

def odd_propagator(h1, hi, hL, tau, L, d):
    """Propagator for odd bond indices (1,..L-1) """

    hs = [hi for _ in range(int(np.floor(L/2)))]
    hL = hL[None,None,:,:] # extend trivial bond dims
    if iseven(L):
        odd_tensors = one_site_from_two_site_tensors(hs, tau, L, d)
    else:
        odd_tensors = one_site_from_two_site_tensors(hs, tau, L, d) + [hL]
    return MPO(odd_tensors)

def even_propagator(h1, hi, hL, tau, L, d):
    """Propagator for even bond indices (1,..L-1) """

    hs = [hi for _ in range(int(np.floor((L-1)/2)))]
    h1, hL = h1[None,None,:,:], hL[None,None,:,:] # extend trivial bond dims
    if iseven(L):
        odd_tensors = [h1] + one_site_from_two_site_tensors(hs, tau, L, d) + [hL]
    else:
        odd_tensors = [h1] + one_site_from_two_site_tensors(hs, tau, L, d)
    return MPO(odd_tensors)

def trotter2(h1, hi, hL, tau, L, d):
    """Second-order Trotter expansion"""

    odd_mpo = odd_propagator(h1, hi, hL, tau/2, L, d)
    even_mpo = even_propagator(h1, hi, hL, tau, L, d)

    return [odd_mpo, even_mpo, odd_mpo]

def trotter4(h1, hi, hL, tau, L, d):
    """Fourth-order Trotter expansion"""

    tau1 = tau/(4-4**(1/3))
    tau2 = tau1
    tau3 = tau - 2*tau1 - 2*tau2

    mpo_tau1 = trotter2(h1, hi, hL, tau1, L, d)
    mpo_tau2 = trotter2(h1, hi, hL, tau2, L, d)
    mpo_tau3 = trotter2(h1, hi, hL, tau3, L, d)

    return [mpo_tau1, mpo_tau2, mpo_tau3, mpo_tau2, mpo_tau1]

def xxz_propagator(L, d, dt, J, Jz, h=0, Dmax=None):
    """Generate sequence of MPOs for one time step.
    Use 4-th order Trotter scheme."""

    Sz = 0.5 * np.array([[1,0],[0,-1]])
    Splus = np.array([[0,1],[0,0]])
    Sminus = np.array([[0,0],[1,0]])

    h1 = np.eye(d,d)
    hL = np.eye(d,d)
    hi = J/2 * np.kron(Splus, Sminus) + J/2 * np.kron(Sminus, Splus) + Jz * np.kron(Sz, Sz)

    return trotter4(h1,hi,hL,dt,L,d)

### Simulation ###

def clear_line(n=1):
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for i in range(n):
        print(LINE_UP, end=LINE_CLEAR)

if __name__ == '__main__':
    T = 20 # simulation time
    steps = 100 # descretization steps of sim. time
    dt = T/steps # sim. time increment

    L = 10 # No. lattice sites
    d = 2 # Hilbert space dimensionality per lattice site
    J = 0.7
    Jz = 1
    h = np.pi / 10
    Dmax = 32 # Max. bond dimension
    Sz = 0.5 * np.array([[1,0],[0,-1]]) # Z-spin matrix

    # MPOs for one step of time evolution
    mpos_list = xxz_propagator(L, d, dt, J, Jz, h, Dmax)

    # Neel state 
    psi = np.zeros([2]*L)
    psi[0,1,0,1,0,1,0,1,0,1] = 1

    # Domain wall state
    psi2 = np.zeros([2]*L)
    psi2[0,0,0,0,0,1,1,1,1,1] = 1

    states = [psi, psi2]
    for i, state in enumerate(states):
        mps = leftcanonical_from_physical(state)

        print(f"--- Start simulation for state {i} ---")
        print(f"Simulation parameters: T={T}, steps={steps}, dt={dt}")
        print(f"XXZ chain, L={L}, d={d}, J={J}, Jz={Jz}, h={h:.2f}")

        exps = []
        ents = []

        # Run simulation loop
        for step in range(steps):
            for mpos in mpos_list:
                for mpo in mpos:
                    mps = mpo * mps
                mps = sweep(mps, Dmax=Dmax) # Renormalize state

            exps.append( expect_all(mps, Sz) )
            ents.append( entropy_all(mps) )

            if step > 0: clear_line(8)
            print(f"""Time step: {step}/{steps} 
    MPS shape (max. bond {Dmax}):
        {mps.shapes}
    Single-site Sz expectation values:
        {'  '.join([f'{exp:.2f}' for exp in exps[-1]])}
    Bond entanglement entropies: 
        {'  '.join([f'{ent:.2f}' for ent in ents[-1]])}
    """)

        fig, ax = plt.subplots(2,1, figsize=(6,8), sharex=True)
        ax[0].plot(exps)
        ax[1].plot(ents)
        ax[1].set_xlabel('step')
        ax[0].set_ylabel('Single-site Z expectation values')
        ax[0].legend(range(L), title='Site index', ncol=5, loc='lower right')
        ax[1].legend(range(L-1), title='bond index', ncol=5, loc='lower right')
        ax[1].set_ylabel(f'Bond entanglement entropies')
        fig.suptitle(f'XXZ chain, $L={L}, d={d}, J={J}, J_z={Jz}, h={h:.2f}, dt={dt}$')
        plt.show()
