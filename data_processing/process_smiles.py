import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
import numpy as np
import pandas as pd
from pandarallel import pandarallel as pdl

class MolecularData(object):

    def get_mol(smiles_or_mol):
        """Loads a molecule from a SMILES string or RDKit Mol object. Adapted from MOSES
        https://github.com/molecularsets/moses

        Args:
            smiles_or_mol (str): Function takes SMILES string.

        Returns:
            None or (RDKit Mol): Function returns RDKit Mol object or None if invalid SMILES.
        """

        if isinstance(smiles_or_mol, str):

            if len(smiles_or_mol) == 0:

                return None
            
            mol = Chem.MolFromSmiles(smiles_or_mol)

            if mol is None:

                return None
            
            try:

                Chem.SanitizeMol(mol)

            except ValueError:

                return None
            
            return mol
        
        return smiles_or_mol
    
    @staticmethod
    def validity(smiles):
        """Checks the validity of a SMILES string.

        Args:
            smiles (str): Function takes SMILES string.

        Returns:
            (None or str): Function returns None if invalid SMILES, otherwise returns SMILES.
        """

        val = Chem.MolToSmiles(MolecularData.get_mol(smiles))

        return val
    
    @staticmethod
    def valid_ratio(smiles_list):
        """Function to get the ratio of valid SMILES strings.

        Args:
            smiles_list (list): Given list of SMILES strings.

        Returns:
            (float): Ratio of valid SMILES strings.
        """

        assert isinstance(smiles_list, list), "Input must be a list of SMILES strings"

        valid = 0

        for smiles in smiles_list:

            if MolecularData.validity(smiles) is not None:

                valid += 1

        print("{} valid SMILES strings found in {} SMILES. {}% Valid".format(valid, len(smiles_list), round(valid/len(smiles_list)*100, 2)))

    
    @staticmethod
    def mol_fingerprints(smiles, chir=True, fpsize=1024, rad=2, feat=False):
        """Generates Morgan fingerprints for a SMILES string.

        Args:
            smiles (str): Given SMILES string.
            chir (bool, optional): Represent molecules as chiral. Defaults to True.
            fpsize (int, optional): Fingerprint size. Defaults to 1024.
            rad (int, optional): Radius size. Defaults to 2.
            feat (bool, optional): Include atomic features. Defaults to False.

        Returns:
            (np.array): Function returns fingerprint as a numpy array.
        """

        assert isinstance(smiles, str), "Input must be a SMILES string"

        mol = MolecularData.get_mol(smiles)

        fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=chir, radius=rad, nBits=fpsize,useFeatures=feat))

        return fp
    
    @staticmethod
    def maccs_fingerprints(smiles):
        """Generates MACCS fingerprints for a SMILES string.

        Args:
            smiles (str): Given SMILES string.

        Returns:
            (np.array): Function returns fingerprint as a numpy array.
        """

        assert isinstance(smiles, str), "Input must be a SMILES string"

        mol = MolecularData.get_mol(smiles)

        fp = np.array(MACCSkeys.GenMACCSKeys(mol))

        return fp

    @staticmethod
    def get_uniques(smiles_list):
        """Function to get unique SMILES strings.

        Args:
            smiles_list (list): Given list of SMILES strings.

        Returns:
            (list): List of unique SMILES strings.
        """

        assert isinstance(smiles_list, list), "Input must be a list of SMILES strings"

        uniques = list(set(smiles_list))

        print("{} unique SMILES strings found in {} SMILES. {}% Unique".format(len(uniques), len(smiles_list), round(len(uniques)/len(smiles_list)*100, 2)))

        return uniques
    
    @staticmethod
    def get_novels(smiles_list, ref_list):
        """Function to get novel SMILES strings.

        Args:
            smiles_list (list): Given list of SMILES strings.
            ref_list (list): Given list of refernce SMILES strings.

        Returns:
            (list): List of novel SMILES strings.
        """

        assert isinstance(smiles_list, list), "Input must be a list of SMILES strings"

        smiles_list = set(smiles_list)

        ref_list = set(ref_list)

        novels = smiles_list - ref_list

        print("{} novel SMILES strings found in {} SMILES. {}% Novel".format(len(novels), len(smiles_list),round(len(novels)/len(smiles_list)*100, 2)))

        return list(novels)
    

class DataHandling(object):

    def bulk_fp(smiles_list, fp_func, worker_num=2, **kwargs):
        """Function to get fingerprints for a list of SMILES strings. 
           If you have large amounts of SMILES strings, this function is faster than using a for loop.
           Uses pandarasallel to parallelize the fingerprint generation.

        Args:
            smiles_list (list): Given list of SMILES strings.
            fp_func (function): Function to generate fingerprints.
            worker_num (int, optional): Number of workers for parallelization. Defaults to 2.
            **kwargs: Keyword arguments for fingerprint function.

        Returns:
            (np.array): Function returns fingerprints as a numpy array.
        """

        assert isinstance(smiles_list, list), "Input must be a list of SMILES strings"

        pdl.initialize(progress_bar=True, nb_workers=worker_num)

        fps = pd.Series(smiles_list).parallel_apply(fp_func, **kwargs).values

        return np.array(fps.tolist())
    


    