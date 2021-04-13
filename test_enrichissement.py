import unittest
import enrichissement
import json
import pickle

# Partie data
with open("config.json") as f:
    conf = json.load(f)
with open('df_nettoye', 'rb') as df_nettoye:
        df = pickle.load(df_nettoye)

class TestEnrichissementMethods(unittest.TestCase):

    def test_enrichissement_geo(self):
        pass

    def test_reorganisation(self):
        pass

    def test_enrichissement_acheteur(self):
        pass

    def test_enrichissement_cpv(self):
        pass

    def test_get_df_enrichissement(self):
        pass

    def test_get_scrap_dataframe(self):
        pass

    def test_get_enrichissement_scrap(self):
        pass

    def test_get_enrichissement_insee(self):
        pass

    def test_get_siretdf_from_original_data(self):
        pass

    def test_enrichissement_siret(self):
        pass

    def test_apply_luhn(self):
        pass

    def test_is_luhn_valid(self):
        pass

    def test_enrichissement_type_entreprise(self):
        pass

    def test_enrichissement_departement(self):
        pass

    def test_extraction_departement_from_code_postal(self):
        pass

    def test_manage_column_final(self):
        pass

    def test_detection_accord_cadre(self):
        pass

if __name__ == '__main__':
    unittest.main()