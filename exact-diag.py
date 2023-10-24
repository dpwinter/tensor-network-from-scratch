import numpy as np
import matplotlib.pyplot as plt
import functools as ft
import scipy

# model params
L = 10
J = 0.7
Jz = 1
h = np.pi / 10

# simulation params
T = 20
steps = 100
dt = T / steps

# constants
up = np.array([1,0])[:,None]
down = np.array([0,1])[:,None]

# multi-site state
psi_neel = up
for i in range(L-1):
    psi_neel = np.kron(psi_neel, down) if i%2==0 else np.kron(psi_neel, up)

psi_wall = up
for i in range(L//2-1):
    psi_wall = np.kron(psi_wall, up)
for i in range(L//2,L):
    psi_wall = np.kron(psi_wall, down)

# single-site operators
Sz = 0.5 * np.array([[1,0],[0,-1]])
Sp = np.array([[0,1],[0,0]])
Sm = np.array([[0,0],[1,0]])
I = np.eye(2)

# pad op with identities
rkron = lambda lst: ft.reduce(np.kron, lst)
def pad(op, i, offset=1):
    if i==0: 
        return rkron( [op] + [I] * (L-offset) )
    return rkron([I] * i + [op] + [I] * (L-i-offset))

# multi-site operators
SzSzs = np.add.reduce([pad(np.kron(Sz,Sz),i,2) for i in range(L-1)])
SpSms = np.add.reduce([pad(np.kron(Sp,Sm),i,2) for i in range(L-1)])
SmSps = np.add.reduce([pad(np.kron(Sm,Sp),i,2) for i in range(L-1)])
Szs = np.add.reduce([pad(Sz,i) for i in range(L)])

# Hamiltonian
H = J/2 * (SpSms + SmSps) + Jz * SzSzs - h * Szs

# propagator
U = scipy.linalg.expm(-1j * H * dt)

# expectation value
expect = lambda op,psi: np.real_if_close( (psi.T.conj() @ op @ psi)[0][0] ).tolist()
expect_all_z = lambda psi: [expect(pad(Sz, i), psi) for i in range(L)]

# simulation
for psi in [psi_neel, psi_wall]:

    exps = []
    for step in range(steps):
        psi = U @ psi
        exp = expect_all_z(psi)
        exps.append(exp)
        print(f"Time step: {step:3d}/{steps}, {np.round(exp,2)}")

    # plot
    plt.plot(exps)
    plt.show()
