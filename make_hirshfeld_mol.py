from pathlib import Path
import sqlite3
import pickle

from rdkit import Chem
from rdkit.Chem import AllChem
from pyscf import dft, M as pyscfMol
from chemprop.utils import make_mol
from pyscf.hirshfeld import HirshfeldAnalysis


# required for atom properties to be pickled
# https://github.com/rdkit/rdkit/issues/1320#issuecomment-280406006
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)


# caching logic
# running QM calculations takes a long time - in the event that something
# crashes we don't want to start from scratch. We'll use these (mostly AI generated)
# functions to handle this for us!
_CACHE_DB = Path(__file__).parent / "_hirshfeld_cache.db"

def _get_cache_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_CACHE_DB)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cache (
            smiles     TEXT NOT NULL,
            basis      TEXT NOT NULL,
            functional TEXT NOT NULL,
            molblob    BLOB NOT NULL,
            PRIMARY KEY (smiles, basis, functional)
        )
        """
    )
    return conn

def _lookup_cache(smiles: str, basis: str, functional: str) -> Chem.Mol | None:
    conn = _get_cache_conn()
    cur = conn.execute(
        "SELECT molblob FROM cache WHERE smiles=? AND basis=? AND functional=?",
        (smiles, basis, functional),
    )
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    try:
        mol = pickle.loads(row[0])
    except Exception:
        # Corrupt entry – delete it
        _delete_cache_entry(smiles, basis, functional)
        return None
    return mol

def _store_cache(smiles: str, basis: str, functional: str, mol: Chem.Mol) -> None:
    molblob = pickle.dumps(mol, protocol=pickle.HIGHEST_PROTOCOL)
    conn = _get_cache_conn()
    conn.execute(
        """
        INSERT OR REPLACE INTO cache (smiles, basis, functional, molblob)
        VALUES (?, ?, ?, ?)
        """,
        (smiles, basis, functional, molblob),
    )
    conn.commit()
    conn.close()

def _delete_cache_entry(smiles: str, basis: str, functional: str) -> None:
    conn = _get_cache_conn()
    conn.execute(
        "DELETE FROM cache WHERE smiles=? AND basis=? AND functional=?",
        (smiles, basis, functional),
    )
    conn.commit()
    conn.close()


# QM requires 3D - we'll make a quick guess with RDKit
# ai generated (gpt-oss:20b run locally) and manually adapted from there
def _embed_3d(mol: Chem.Mol):
    # Generate 3‑D coordinates with ETKDG
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    result = AllChem.EmbedMolecule(mol, params)
    if result != 0:
        raise RuntimeError(f"Unable to embed molecule {Chem.MolToSmiles(mol)}")
    # Optional: optimize geometry
    AllChem.UFFOptimizeMolecule(mol)

# get the atoms at XYZ format
# https://github.com/rdkit/rdkit/issues/7716#issue-2466027153
def _rdkit_to_pyscf(mol: Chem.Mol):
    atom_str = ''
    conf = mol.GetConformer()
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        atom_str += f'{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}; '
    return atom_str


def _compute_charge_and_spin(mol: Chem.Mol) -> tuple[int, int]:
    """
    Compute the formal charge and the minimal spin multiplicity
    that makes the electron count even.

    Returns
    -------
    charge : int
        Formal charge of the molecule (default 0).
    spin : int
        0 for singlet, 1 for doublet, 2 for triplet, …
    """
    # Formal charge (RDKit can give it directly)
    charge = Chem.GetFormalCharge(mol)

    # Total number of electrons = sum atomic numbers – charge
    nelec = sum(atom.GetAtomicNum() for atom in mol.GetAtoms()) - charge

    # If the number of electrons is odd, we need a doublet (spin=1)
    spin = 0 if nelec % 2 == 0 else 1
    return charge, spin


def make_hirshfeld_mol(smiles: str, basis: str = '6-31G*', dft_functional: str = 'B3LYP', use_gpu: bool = False):
    cached = _lookup_cache(smiles, basis, dft_functional)
    if cached is not None:
        return cached
    mol = make_mol(
        smiles,
        keep_h=True,
        add_h=True,
        ignore_stereo=False,
        reorder_atoms=False,
    )
    _embed_3d(mol)
    charge, spin = _compute_charge_and_spin(mol)
    pyscf_mol = pyscfMol(
        atom=_rdkit_to_pyscf(mol),
        verbose=1,
        basis=basis,
        spin=spin,
        charge=charge,
    )
    mean_field = dft.RKS(
        pyscf_mol,
        xc=dft_functional,
    )
    if use_gpu:
        try:
            mean_field = mean_field.to_gpu()
        except ImportError as e:  # gpu4pyscf is probably not installed, or not installed properly (e.g., libcusolver.so.11 may be missing if nvidia-cuda-toolkit is missing)
            raise RuntimeError("`gpu4pyscf` is either not installed or not installed correctly, install it or set `use_gpu=False`") from e
        except Exception as e:  # gpu4pyscf is installed properly, but no GPU was detected (or other error)
            if "no CUDA-capable device is detected" in str(e):
                raise RuntimeError("GPU not detected, ensure it is visible or if none is present set `use_gpu=False`") from e
            else:
                raise RuntimeError(f"Unexpected error occurred during simulation of molecule `{smiles}`!") from e
    mean_field.run()

    if use_gpu:
        import cupy

        # not all components are preserved on this operation - manually carry some around
        mo_coeff  = cupy.asnumpy(mean_field.mo_coeff)
        mo_occ    = cupy.asnumpy(mean_field.mo_occ)
        mo_energy = cupy.asnumpy(mean_field.mo_energy)
        dm        = cupy.asnumpy(mean_field.make_rdm1())

        # build a fresh CPU mean-field object
        mf_cpu = dft.RKS(mean_field.mol, xc=mean_field.xc)
        mf_cpu.grids.build()

        # copy over results
        mf_cpu.mo_coeff  = mo_coeff
        mf_cpu.mo_occ    = mo_occ
        mf_cpu.mo_energy = mo_energy
        mf_cpu.e_tot     = mean_field.e_tot  # total energy
        mf_cpu.converged = mean_field.converged

        # set density matrix explicitly so Hirshfeld can use it
        mf_cpu._dm = dm
        mean_field = mf_cpu


    # post-hoc hirshfeld analysis for each atom
    H = HirshfeldAnalysis(
        mean_field,
    ).run()
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetDoubleProp("charge_eff", H.result["charge_eff"][i])
        atom.SetDoubleProp("dipole_eff_x", H.result["dipole_eff"][i][0])
        atom.SetDoubleProp("dipole_eff_y", H.result["dipole_eff"][i][1])
        atom.SetDoubleProp("dipole_eff_z", H.result["dipole_eff"][i][2])
        atom.SetDoubleProp("V_eff", H.result["V_eff"][i])
        atom.SetDoubleProp("V_free", H.result["V_free"][i])
    _store_cache(smiles, basis, dft_functional, mol)
    return mol

# Example usage
if __name__ == "__main__":
    mol_cpu = make_hirshfeld_mol("C", basis="sto3g", dft_functional="lda", use_gpu=True)
    Path("_hirshfeld_cache.db").unlink()
    mol_gpu = make_hirshfeld_mol("C", basis="sto3g", dft_functional="lda", use_gpu=False)
    mol_gpu = make_hirshfeld_mol("C", basis="sto3g", dft_functional="lda", use_gpu=False)  # cache test
    assert abs(mol_gpu.GetAtomWithIdx(1).GetDoubleProp("V_eff") - mol_cpu.GetAtomWithIdx(1).GetDoubleProp("V_eff")) < 0.000_001
