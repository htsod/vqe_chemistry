import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-mol", help="input molecular data", type=str, default="h4.csv")
parser.add_argument("-x", "--output-mol", help="output molecular data", type=str, default="h4_best.csv")
args = parser.parse_args()


import numpy as np
import quantumsymmetry
from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
import mindspore as ms
from mindquantum.core.operators import InteractionOperator, normal_ordered
from mindquantum.algorithm.nisq import uccsd_singlet_generator
from mindquantum.core.circuit import Circuit
from mindquantum.simulator import Simulator
from mindquantum.core.operators import FermionOperator, Hamiltonian
from mindquantum.framework import MQAnsatzOnlyLayer
from scipy.optimize import minimize
import time

from mindquantum.core.parameterresolver import ParameterResolver
from mindquantum.core.operators import QubitOperator



def from_openfermion(of_ops) -> "QubitOperator":
    """
    Convert qubit operator from openfermion to mindquantum format.

    Args:
        of_ops (openfermion.QubitOperator): Qubit operator from openfermion.

    Returns:
        QubitOperator, qubit operator from mindquantum.
    """
    # pylint: disable=import-outside-toplevel
    try:
        from openfermion import QubitOperator as OFQubitOperator
    except (ImportError, AttributeError):
        _require_package("openfermion", "1.5.0")
    if not isinstance(of_ops, OFQubitOperator):
        raise TypeError(
            "of_ops should be a QubitOperator" f" from openfermion framework, but get type {type(of_ops)}"
        )
    out = QubitOperator()
    for term, v in of_ops.terms.items():
        out += QubitOperator(' '.join([f"{j}{i}" for i, j in term]), ParameterResolver(v))
    return out




# read csv file and put molecular formula into list and array
def read_csv(file_name):
    with open(file_name, 'r') as f:
        data = f.readlines()
    mol_name = []
    mol_poi = []
    for i in data:
        tmp = i.split(',')
        mol_name.append(tmp[0])
        mol_poi.extend([float(i) for i in tmp[1:]])
    return mol_name, np.array(mol_poi)


def gene_uccsd(mol):
    geometry = mol[1].reshape(len(mol[0]), -1)
    geometry = [[mol[0][i], list(j)] for i, j in enumerate(geometry)]
    basis = "sto3g"
    molecule_of = MolecularData(geometry, basis, multiplicity=1, data_directory='./')
    mol = run_pyscf(
        molecule_of,
        run_fci=1,
    )
    try:
        r_ham = quantumsymmetry.reduced_hamiltonian(
        atom = mol.geometry,
        basis = 'sto-3g',
        verbose = True,
        output_format = 'openfermion'
        )
            ucc_fermion_ops = uccsd_singlet_generator(
        mol.n_qubits, mol.n_electrons, anti_hermitian=True)
        ucc_qubit_ops = Transform(ucc_fermion_ops).jordan_wigner()

        hartreefock_wfn_circuit = Circuit([X.on(i) for i in range(mol.n_electrons)])
        ansatz_circuit = TimeEvolution(ucc_qubit_ops.imag, 1.0).circuit
        ansatz_parameter_names = ansatz_circuit.params_name
        total_circuit = hartreefock_wfn_circuit + ansatz_circuit
    except:
        ham_of = mol.get_molecular_hamiltonian()
        inter_ops = InteractionOperator(*ham_of.n_body_tensors.values())
        ham_hiq = FermionOperator(inter_ops)
        ham_fo = normal_ordered(ham_hiq).real
        ham = ESConserveHam(ham_fo)
        ucc_fermion_ops = uccsd_singlet_generator(mol.n_qubits, mol.n_electrons, anti_hermitian=False)
        total_circuit = Circuit()
        for term in ucc_fermion_ops:
            total_circuit += ExpmPQRSFermionGate(term)

    ham = from_openfermion(r_ham)

    


    return ham, total_circ, mol.n_qubits, mol.n_electrons


def run_uccsd(ham, circ, nq, ne):
    sim = Simulator('mqvector', circ.n_qubits)
    grad_ops = sim.get_expectation_with_grad(Hamiltonian(ham), circ)

    molecule_pqcnet = MQAnsatzOnlyLayer(grad_ops, 'Zeros')
    initial_energy = molecule_pqcnet()
    print("Initial energy: %20.16f" % (initial_energy.asnumpy()))

    optimizer = ms.nn.Adagrad(molecule_pqcnet.trainable_params(), learning_rate=4e-2)
    train_pqcnet = ms.nn.TrainOneStepCell(molecule_pqcnet, optimizer)
    

    eps = 1.e-8
    energy_diff = eps * 1000
    energy_last = initial_energy.asnumpy() + energy_diff
    iter_idx = 0
    while abs(energy_diff) > eps:
        energy_i = train_pqcnet().asnumpy()
        if iter_idx % 5 == 0:
            print("Step %3d energy %20.16f" % (iter_idx, float(energy_i)))
        energy_diff = energy_last - energy_i
        energy_last = energy_i
        iter_idx += 1
    return molecule_pqcnet.weight.asnumpy()


def opti_geo(geo, mol_name):
    ham, circ, nq, ne = gene_uccsd([mol_name, geo])
    res = run_uccsd(ham, circ, nq, ne)
    #print(res,'\t', time.ctime())
    best_x = geo.reshape(len(mol_name), -1)
    out = []
    for idx, n in enumerate(mol_name):
        tmp = [n]
        tmp.extend([str(i) for i in best_x[idx]])
        out.append(', '.join(tmp) + '\n')

    with open(args.output_mol, 'w') as f:
        f.writelines(out)

    return res

name, p0 = read_csv(args.input_mol)
p0=np.random.uniform(size=len(p0))
res = minimize(opti_geo, p0, args=(name, ), method='BFGS')
best_x = res.x.reshape(len(name), -1)
out = []
for idx, n in enumerate(name):
    tmp = [n]
    tmp.extend([str(i) for i in best_x[idx]])
    out.append(', '.join(tmp) + '\n')

with open(args.output_mol, 'w') as f:
    f.writelines(out)


