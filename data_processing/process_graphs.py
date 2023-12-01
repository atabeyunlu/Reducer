# Add sparse graph processing and dense graph processing. It should be able to convert from SMILES or to SMILES.
from rdkit import Chem
import numpy as np

class DenseGraphs(object):

    def one_of_k_encoding_unk(self, x, allowable_set):

        """Maps inputs not in the allowable set to the last element. Taken from:
        https://github.com/HIPS/neural-fingerprint/blob/master/neuralfingerprint/util.py"""

        if x not in allowable_set:

            x = allowable_set[-1]

        return map(lambda s: x == s, allowable_set)

    def __init__(self, smiles_list, max_atoms=50, max_degree=10, max_valence=4, max_ring_size=6, min_ring_size=3):
        """
        :param smiles_list: list of SMILES strings
        :param max_atoms: maximum number of atoms in a molecule
        :param max_degree: maximum node degree
        :param max_valence: maximum valence
        :param max_ring_size: maximum ring size
        :param min_ring_size: minimum ring size
        """
        self.smiles_list = smiles_list
        self.max_atoms = max_atoms
        self.max_degree = max_degree
        self.max_valence = max_valence
        self.max_ring_size = max_ring_size
        self.min_ring_size = min_ring_size

    def get_adjacency_tensor(self):
        """
        :return: adjacency tensor of shape (num_molecules, max_atoms, max_atoms, max_degree)
        """
        adj = np.zeros((len(self.smiles_list), self.max_atoms, self.max_atoms, self.max_degree))
        for i, smiles in enumerate(self.smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            adj[i, :, :, :] = Chem.GetAdjacencyMatrix(mol, useBO=True)
        return adj

    def get_node_features(self):
        """
        :return: node features of shape (num_molecules, max_atoms, atom_types)
        """
        node_features = np.zeros((len(self.smiles_list), self.max_atoms, 34))
        for i, smiles in enumerate(self.smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            for atom in mol.GetAtoms():
                node_features[i, atom.GetIdx(), :] = self.one_of_k_encoding_unk(atom.GetSymbol(),
                                                                           ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Mg',
                                                                            'Na', 'Br', 'Fe', 'Ca', 'Cu', 'Mc', 'Pd',
                                                                            'Pb', 'K', 'I', 'Al', 'Ni', 'Mn', 'Zn',
                                                                            'Se', 'Si', 'As', 'B', 'V', 'H', 'Li',
                                                                            'Sn', 'Ag', 'Co', 'Cd', 'Ge', 'Hg', 'Mo',
                                                                            'Unknown'], 34)
                
        return node_features
    
    def densegraph_to_smiles (self, adj, node_features):

        """
        :param adj: adjacency tensor of shape (num_molecules, max_atoms, max_atoms, max_degree)
        :param node_features: node features of shape (num_molecules, max_atoms, atom_types)
        :return: smiles_list: list of SMILES strings
        """

        smiles_list = []

        for i in range(adj.shape[0]):
            mol = Chem.RWMol()

            for j in range(adj.shape[1]):
                mol.AddAtom(Chem.Atom(node_features[i,j,:]))

            for j in range(adj.shape[1]):

                for k in range(adj.shape[3]):
                    if adj[i,j,:,k] == 1:
                        mol.AddBond(j, k, Chem.BondType.SINGLE)

            smiles_list.append(Chem.MolToSmiles(mol))

        return smiles_list
    
    def sparsegraph_to_smiles(x):
        