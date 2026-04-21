import os
import tempfile
import unittest

import numpy as np

from ecd_platform.config import CDGauge, ECDConfig, QMProgram
from ecd_platform.conformer import ConformerRecord, ConformerStatus
from ecd_platform.gaussian_parser import extract_cd_data, extract_energies
from ecd_platform.parser_dispatch import parse_single_file, same_output_file


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAMPLE_LOG = os.path.join(ROOT, "ecd_results", "21gaussian", "21-RR_1.log")


GAUSSIAN_TD_SNIPPET = """
 Entering Gaussian System
 SCF Done:  E(RCAM-B3LYP) =  -2068.11723196D+00     A.U. after   14 cycles
 Excitation energies and oscillator strengths:
 Excited State   1:      Singlet-A      4.8879 eV  253.65 nm  f=0.0410
 Excited State   2:      Singlet-A      4.9225 eV  251.87 nm  f=0.0652
  1/2[<0|del|b>*<b|r|0> + (<0|r|b>*<b|del|0>)*]
       state          XX          YY          ZZ     R(length)
         1      -170.3507    -16.1805    201.2363      4.9017
         2       -41.8525    149.9660    172.5546     93.5560

 Normal termination of Gaussian 16
"""


GAUSSIAN_VELOCITY_ONLY = """
 Excited State   1:      Singlet-A      3.1000 eV  399.95 nm  f=0.0100
 Rotatory Strengths (R) in cgs (10**-40 erg-esu-cm/Gauss)
       state          XX          YY          ZZ    R(velocity)    E-M Angle
         1         1.0000      2.0000      3.0000      4.5000       89.13

"""


class GaussianParserTests(unittest.TestCase):
    def test_extract_scf_energy_accepts_scientific_notation(self):
        rec = ConformerRecord(1, "conf-1")
        extract_energies(GAUSSIAN_TD_SNIPPET, rec)
        self.assertAlmostEqual(rec.scf_energy, -2068.11723196)
        self.assertEqual(rec.errors, [])

    def test_extract_excited_states_and_length_rotatory_strengths(self):
        rec = ConformerRecord(1, "conf-1")
        ok = extract_cd_data(GAUSSIAN_TD_SNIPPET, rec, CDGauge.LENGTH)
        self.assertTrue(ok)
        np.testing.assert_allclose(rec.transition_energies, [4.8879, 4.9225])
        np.testing.assert_allclose(rec.rotatory_strengths, [4.9017, 93.5560])
        self.assertEqual(rec.n_transitions, 2)

    def test_length_gauge_falls_back_to_velocity(self):
        rec = ConformerRecord(1, "conf-1")
        ok = extract_cd_data(GAUSSIAN_VELOCITY_ONLY, rec, CDGauge.LENGTH)
        self.assertTrue(ok)
        np.testing.assert_allclose(rec.rotatory_strengths, [4.5])
        self.assertTrue(any("falling back to 'velocity'" in w for w in rec.warnings))

    @unittest.skipUnless(os.path.exists(SAMPLE_LOG), "sample Gaussian log is absent")
    def test_sample_gaussian_file_auto_parses(self):
        rec = ConformerRecord(1, "conf-1")
        parse_single_file(SAMPLE_LOG, rec, ECDConfig(program=QMProgram.AUTO))
        self.assertEqual(rec.status, ConformerStatus.OK)
        self.assertAlmostEqual(rec.scf_energy, -2068.11723196)
        self.assertEqual(rec.n_transitions, 30)
        self.assertEqual(rec.errors, [])
        self.assertLessEqual(len(rec.warnings), 1)

    @unittest.skipUnless(os.path.exists(SAMPLE_LOG), "sample Gaussian log is absent")
    def test_explicit_orca_rejects_gaussian_file_without_orca_errors(self):
        rec = ConformerRecord(1, "conf-1")
        parse_single_file(SAMPLE_LOG, rec, ECDConfig(program=QMProgram.ORCA))
        self.assertEqual(rec.status, ConformerStatus.PARSE_FAILED)
        self.assertTrue(any("QM program mismatch" in e for e in rec.errors))
        self.assertFalse(any("CD SPECTRUM" in e for e in rec.errors))

    def test_same_output_file_handles_normalized_path_strings(self):
        with tempfile.NamedTemporaryFile(suffix=".log") as tmp:
            path_a = tmp.name
            path_b = os.path.abspath(path_a).replace(os.sep, "/")
            self.assertTrue(same_output_file(path_a, path_b))


if __name__ == "__main__":
    unittest.main()
