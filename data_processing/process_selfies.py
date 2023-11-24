import selfies as sf

class ProcessSelfies:

    def selfies_to_smiles(selfies):
        """Converts SELFIES string to SMILES string.

        Args:
            selfies (str): SELFIES string.

        Returns:
            (str): SMILES string.
        """

        smiles = sf.decoder(selfies)

        return smiles
    
    def smiles_to_selfies(self, smiles):
        """Converts SMILES string to SELFIES string.

        Args:
            smiles (str): SMILES string.

        Returns:
            (str): SELFIES string.
        """

        selfies = sf.encoder(smiles)

        return selfies