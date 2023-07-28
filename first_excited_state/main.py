# Here is an example of orthogonally constrained VQE (OC-VQE) method
# calculating H2 first excited state energy by MindQuantum.

# Reference: arXiv:1805.08138

import numpy as np
import sys
from mindquantum.core.gates import X, RY, RZ
from mindquantum.core.circuit import Circuit
from mindquantum.simulator import Simulator
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.algorithm.nisq import generate_uccsd
from mindquantum.algorithm.nisq import get_qubit_hamiltonian, HardwareEfficientAnsatz
from openfermionpyscf import run_pyscf
from openfermion.chem import MolecularData
from scipy.optimize import minimize

# Define molecular geometry and basis set
# geometry = [
#     ["H", [0.0, 0.0, 0.0]],
#     ["H", [0.0, 0.0, 1.4]],
# ]
# basis = "sto3g"
# spin = 0
# print("Geometry: \n", geometry)

# Initialize molcular data
# mol = MolecularData(geometry, basis, multiplicity=2 * spin + 1)

# mol = MolecularData(geometry, basis, multiplicity=2 * spin + 1, data_directory=sys.path[0])
# mol = run_pyscf(mol)

# molecules = [
#     ("H4_0.5", -0.90886229),
#     ("H4_1.0", -1.93375723),
#     ("H4_1.5", -1.92555851),
#     ("LiH_0.5", -7.00849729),
#     ("LiH_1.5", -7.7606092),
#     ("LiH_2.5", -7.77815121),
# #   Unknown1,
# #   Unknown2,
# #   Unknown3,
# ]


# f_index = 0
# fname = molecules[f_index][0] + ".hdf5"
# mol = MolecularData(filename=fname)
# print(f" number of electrons = {mol.n_electrons}")
# print(f" fci energy of {fname} = {mol.fci_energy}")


beta = 0.8
options = {"disp": True, "gtol":1e-6}


# Construct hartreefock wave function circuit




def hf_circ(mol):
    hartreefock_wfn_circuit = Circuit([X.on(i) for i in range(mol.n_electrons)])
    return hartreefock_wfn_circuit




def setup_gs(mol):
    hartreefock_wfn_circuit = hf_circ(mol)

    # Get hamiltonian of the molecule
    ham_op = get_qubit_hamiltonian(mol)
    # Construct ground state ansatz circuit
    gs_circ = (
        hartreefock_wfn_circuit
        + HardwareEfficientAnsatz(mol.n_qubits, [RY, RZ], depth=5).circuit
    )
    return ham_op, gs_circ

# Define the objective function to be minimized
def gs_func(x, grad_ops):
    f, g = grad_ops(x)
    return np.real(np.squeeze(f)), np.real(np.squeeze(g))


def train_gs_layer(mol):
    ham_op, gs_circ = setup_gs(mol)

    # Declare the ground state simulator
    gs_sim = Simulator("mqvector", gs_circ.n_qubits)

    # Get the expectation and gradient calculating function
    gs_grad_ops = gs_sim.get_expectation_with_grad(Hamiltonian(ham_op), gs_circ)

    # Initialize amplitudes
    init_amplitudes = np.random.random(len(gs_circ.all_paras))

    # Get Optimized result
    gs_res = minimize(gs_func, init_amplitudes, args=(gs_grad_ops), method="bfgs", jac=True, options=options)

    # Construct parameter resolver of the ground state circuit
    gs_pr = dict(zip(gs_circ.params_name, gs_res.x))
    
    return gs_pr, ham_op, gs_sim, gs_circ



def setup_es(mol):
    hartreefock_wfn_circuit = hf_circ(mol)
    gs_pr, ham_op, gs_sim, gs_circ = train_gs_layer(mol)


    # Evolve into ground state
    gs_sim.apply_circuit(gs_circ, gs_pr)

    # Calculate energy of ground state
    gs_en = gs_sim.get_expectation(Hamiltonian(ham_op)).real

    # -----------------------------------------------------------------

    # Construct excited state ansatz circuit
    es_circ = (
        hartreefock_wfn_circuit
        + HardwareEfficientAnsatz(mol.n_qubits, [RY, RZ], depth=5).circuit
    )

    return es_circ, ham_op, gs_sim


# Define the objective function to be minimized
def es_func(x, es_grad_ops, ip_grad_ops, beta):
    f0, g0 = es_grad_ops(x)
    f1, g1 = ip_grad_ops(x)
    # Remove extra dimension of the array
    f0, g0, f1, g1 = (np.squeeze(f0), np.squeeze(g0), np.squeeze(f1), np.squeeze(g1))
    cost = np.real(f0) + beta * np.abs(f1) ** 2  # Calculate cost function
    punish_g = np.conj(g1) * f1 + g1 * np.conj(f1)  # Calculate punishment term gradient
    total_g = g0 + beta * punish_g
    return cost, total_g.real



def train_es_layer(mol):
    es_circ, ham_op, gs_sim = setup_es(mol)

    # Declare the excited state simulator
    es_sim = Simulator("mqvector", mol.n_qubits)

    # Get the expectation and gradient calculating function
    es_grad_ops = es_sim.get_expectation_with_grad(Hamiltonian(ham_op), es_circ)

    # Get the expectation and gradient calculating function of inner product
    ip_grad_ops = es_sim.get_expectation_with_grad(
        Hamiltonian(QubitOperator("")), es_circ, Circuit(), simulator_left=gs_sim
    )

    # Initialize amplitudes
    init_amplitudes = np.random.random(len(es_circ.all_paras))

    # Get Optimized result
    es_res = minimize(
        es_func,
        init_amplitudes,
        args=(es_grad_ops, ip_grad_ops, beta),
        method="bfgs",
        jac=True,
        options=options
    )

    # Construct parameter resolver of the excited state circuit
    es_pr = dict(zip(es_circ.params_name, es_res.x))

    # Evolve into excited state
    es_sim.apply_circuit(es_circ, es_pr)

    return es_sim, ham_op


def measurement(mol):
    es_sim, ham_op = train_es_layer(mol)
# Calculate energy of excited state
    es_en = es_sim.get_expectation(Hamiltonian(ham_op)).real
    return es_en


class Main:
    def run(self, mol):
        return measurement(mol)
