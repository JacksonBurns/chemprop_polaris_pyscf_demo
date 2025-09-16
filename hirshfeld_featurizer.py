import numpy as np
from rdkit.Chem.rdchem import Atom
from chemprop.featurizers import VectorFeaturizer


class HirshfeldAtomFeaturizer(VectorFeaturizer[Atom]):
    def __len__(self) -> int:
        return 8

    def __call__(self, a: Atom | None) -> np.ndarray:
        if a is None:
            return np.zeros(self.__len__())
        return np.array(
            [
                a.GetAtomicNum() / 100,
                a.GetTotalNumHs(),
                a.GetDoubleProp("charge_eff"),
                a.GetDoubleProp("V_eff") / 100,
                a.GetDoubleProp("V_free") / 100,
                a.GetDoubleProp("V_eff") / a.GetDoubleProp("V_free"),
                # count of heavy atom neighbors
                sum(1 for _a in a.GetNeighbors() if _a.GetAtomicNum() > 1),
                # dipole magnitude
                np.sqrt(a.GetDoubleProp("dipole_eff_x")**2 +
                    a.GetDoubleProp("dipole_eff_y")**2 +
                    a.GetDoubleProp("dipole_eff_z")**2),
            ]
        )

if __name__ == "__main__":
    from make_hirshfeld_mol import make_hirshfeld_mol

    mol = make_hirshfeld_mol("C", basis="sto3g", dft_functional="pbe0", use_gpu=True)
    feat = HirshfeldAtomFeaturizer()
    print(feat(mol.GetAtomWithIdx(0)))
