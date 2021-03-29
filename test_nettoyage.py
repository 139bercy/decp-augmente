import unittest
import json
import os
import pandas as pd
import nettoyage
import numpy as np
from pandas import json_normalize


# Partie recuperation de données.
with open("config.json") as f:
    conf = json.load(f)
path_to_data = conf["path_to_data"]
decp_file_name = conf["decp_file_name"]
with open(os.path.join(path_to_data, decp_file_name), encoding='utf-8') as json_data:
    data = json.load(json_data)
df = json_normalize(data['marches'])


class TestNetoyageMethods(unittest.TestCase):
    """Classe permettant de tester le module nettoyage"""

    def test_manage_titulaires(self):
        df_input = df[:1000].reset_index()
        taille_input = df_input.shape
        nb_ligne_nulle = sum(df_input.titulaires.isnull())
        df_output = nettoyage.manage_titulaires(df_input)
        nb_ligne_finale = sum(df_input.titulaires.apply(len)) - nb_ligne_nulle + 1
        # test sur la taille finale
        # Pour avoir le nombre de ligne finale, il faut connaitre le nombre de ligne ou titulaires est nulle
        taille_output = df_output.shape
        self.assertEqual(nb_ligne_finale, taille_output[0])  # la taille finale correspond à la somme des len de titulaires
        self.assertEqual(taille_output[1], taille_input[1] - 9 + 3)  # Ajout de 3 colonnes
        # test sur le contenu
        # Premier test sur les nouvelles colonnes
        compteur = 0
        for i in range(len(df_input.titulaires)):
            if df_input.titulaires[i] == '0':
                compteur -= 1
            else:
                nb_tit = 0
                for j in range(len(df_input.titulaires[i])):
                    nb_tit += 1  # valeur nbTitulairesSurCeMarche théorique = jmax
                    if j >= 1:
                        compteur += 1
                    valeur_initiale = df_input.titulaires[i][j]
                    self.assertEqual(valeur_initiale['typeIdentifiant'], df_output.typeIdentifiant[i + compteur])
                    self.assertEqual(valeur_initiale['id'], df_output.idTitulaires[i + compteur])
                    self.assertEqual(valeur_initiale['denominationSociale'], df_output.denominationSociale[i + compteur])

        def test_manage_duplicates(self):
            ligne = df[:5]  # 50563 50547
            ligne2 = df[:2]
            df_input = ligne.append(ligne2).reset_index()  # il y a donc deux doublons de lignes
            df_input = df_input.append(df[df.procedure == 'Appel d’offres restreint']).reset_index()
            df_input = df_input.append(df[50563:50564])
            df_input = nettoyage.manage_titulaires(df_input)  # Fonction testée
            nb_ligne, nb_col = df_input.shape
            nb_ligne_finale = nb_ligne - 2 + 2  # 2 doublons et 2 duplications de lignes
            df_output = nettoyage.manage_duplicates(df_input)
            nb_ligne_output, nb_col_output = df_output.shape
            # Test sur les tailles des entrées/sorties
            self.assertEqual(nb_ligne_finale, nb_ligne_output)
            self.assertEqual(nb_col, nb_col_output)
            # Test sur les 3 np.where
            # On vérifie que les formePrix sont correctement modifiés
            nb_formePrix_a_modifier = len(df_input.formePrix[df_input.formePrix == "Ferme, actualisable"])
            nb_formePrix_restant_a_modifier = len(df_output.formePrix[df_input.formePrix == "Ferme, actualisable"])
            nb_formePrix_modifie = len(df_output.formePrix[df_input.formePrix == 'Ferme et actualisable'])
            self.assertEqual(nb_formePrix_a_modifier, nb_formePrix_modifie)
            self.assertEqual(0, nb_formePrix_restant_a_modifier)

    def test_is_false_amount(self):
        L = [567893256, 444444444, 3456.55555, 55555645.678, 55655655, 10000000]  # G, F, G, F, G
        L_attendu = [False, True, False, True, True, False]
        for i in range(len(L)):
            self.assertEqual(L_attendu[i], nettoyage.is_false_amount(L[i]))

    def test_manage_amount(self):
        df_test = pd.DataFrame([
                               [1],
                               [2000000000],
                               [56789],
                               [555555]],
                               index=[0, 1, 2, 3],
                               columns=['montant'])
        df_attendu = pd.DataFrame([
                                  [0, 1],
                                  [0, 2000000000],
                                  [56789, 56789],
                                  [0, 555555]],
                                  index=[0, 1, 2, 3],
                                  columns=['montant', 'montantOriginal'])
        df_output = nettoyage.manage_amount(df_test)
        # On vérifie si les montant originaux correspondent bien aux montant initiaux
        self.assertEqual(sum(df_output.montantOriginal == df_attendu.montantOriginal), len(df_test))
        # On vérifie si les montants calculés sont effectivement bien calculés
        self.assertEqual(sum(df_output.montant == df_attendu.montant), len(df_test))
        # On vérifie la taille
        self.assertEqual(df_attendu.shape[0], df_output.shape[0])
        self.assertEqual(df_attendu.shape[1], df_output.shape[1])
    
    def test_manage_missing_code(self):
        df_test = pd.DataFrame([
                               ['AETYHGF', np.nan, 'SIRET', "123456789", "Fournitures", "Coucou"],
                               [np.nan, '46562345', 'SIRET', "4444455555", "Fournitures", np.nan]],
                               index=[0, 1],
                               columns=['id', 'codeCPV', 'typeIdentifiant', 'idTitulaires', 'natureObjet', 'denominationSociale'])
        df_attendu = pd.DataFrame([
                                  ['AETYHGF', '00000000', 'SIRET', "123456789", "Fournitures", "Coucou", "56789", "00"],
                                  ['0000000000000000', '46562345', 'SIRET', "4444455555", 'Services', np.nan, "55555", "46"]],
                                  index=[0, 1],
                                  columns=['id', 'codeCPV', 'typeIdentifiant', 'idTitulaires', 'natureObjet', 'denominationSociale', 'nic', "CPV_min"])
        df_output = nettoyage.manage_missing_code(df_test)
        for column in df_attendu:
            df_attendu[column] == df_output[column]



if __name__ == '__main__':
    unittest.main()
