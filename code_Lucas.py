# nettoyage.py


def data_inputation(df):
    # Utilisation de la méthode 5 pour estimer les valeurs manquantes
    df['Region'] = df['Region'].astype(str)
    df['formePrix'] = df['formePrix'].astype(str)
    df['codeCPV'] = df['codeCPV'].astype(str)

    df['moisNotification'] = df['moisNotification'].astype(str)
    df['anneeNotification'] = df['anneeNotification'].astype(str)
    df['conca'] = df['formePrix'] + df['Region'] + df['codeCPV']

    # Calcul de la médiane par stratification
    medianeRegFP = pd.DataFrame(df.groupby('conca')['montant'].median())
    medianeRegFP.reset_index(level=0, inplace=True)
    medianeRegFP.columns = ['conca', 'montantEstimation']
    df = pd.merge(df, medianeRegFP, on='conca', copy=False)
    # Remplacement des valeurs manquantes par la médiane du groupe
    df['montant'] = np.where(df['montant'].isnull(), df['montantEstimation'], df['montant'])
    del df['conca'], df['montantEstimation']

    # On recommence avec une plus petite stratification
    df['conca'] = df['formePrix'] + df['Region']
    df.reset_index(level=0, inplace=True)
    # Calcul de la médiane par stratification
    medianeRegFP = pd.DataFrame(df.groupby('conca')['montant'].median())
    medianeRegFP.reset_index(level=0, inplace=True)
    medianeRegFP.columns = ['conca', 'montantEstimation']
    df = pd.merge(df, medianeRegFP, on='conca', copy=False)
    # Remplacement des valeurs manquantes par la médiane du groupe
    df['montant'] = np.where(df['montant'].isnull(), df['montantEstimation'], df['montant'])
    # S'il reste encore des valeurs nulles...
    df['montant'] = np.where(df['montant'].isnull(), df['montant'].median(), df['montant'])
    del df['conca'], df['montantEstimation'], df['index']
    del medianeRegFP

    # Colonne par marché
    df['montantTotalMarché'] = df["montant"] * df["nbTitulairesSurCeMarche"]

    return df



# enrichissement.py 
import urllib
import folium
import matplotlib.pyplot as plt
from folium import plugins
from folium.plugins import HeatMap
from folium.plugins import MarkerCluster
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler


def getArchiveErrorSIRET():
    """
    ######################################################################
    ### On supprime les siret déjà identifié comme faux
    path = os.path.join(path_to_data, error_siret_file)

    try:
        archiveErrorSIRET = pd.read_csv(path,
                                        sep=';',
                                        encoding='utf-8',
                                        dtype={
                                            'siren': str,
                                            'siret': str,
                                            'denominationSociale': str})
        dfSIRET = pd.merge(dfSIRET, archiveErrorSIRET, how='outer', indicator='source')
        dfSIRET = dfSIRET[dfSIRET.source.eq('left_only')].drop('source', axis=1)
        dfSIRET.reset_index(inplace=True, drop=True)
        print('Erreurs archivées non trouvées')
    except:
        archiveErrorSIRET = pd.DataFrame(columns=['siret', 'siren', 'denominationSociale'])
        print('Aucune archive d\'erreur')
    """
    archiveErrorSIRET = pd.DataFrame(columns=['siret', 'siren', 'denominationSociale'])
    print('Aucune archive d\'erreur')
    return archiveErrorSIRET


def get_enrichissement_insee(dfSIRET, path_to_data):
    # dans StockEtablissement_utf8, il y a principalement : siren, siret, nom établissement, adresse, activité principale

    path = os.path.join(path_to_data, conf["base_sirene_insee"])
    columns = [
        'siren',
        'nic',
        'siret',
        'typeVoieEtablissement',
        'libelleVoieEtablissement',
        'codePostalEtablissement',
        'libelleCommuneEtablissement',
        'codeCommuneEtablissement',
        'activitePrincipaleEtablissement',
        'nomenclatureActivitePrincipaleEtablissement']
    dtypes = {
        'siret': 'string',
        'typeVoieEtablissement': 'string',
        'libelleVoieEtablissement': 'string',
        'codePostalEtablissement': 'string',
        'libelleCommuneEtablissement': 'string',
    }

    result = pd.DataFrame(columns=columns)
    chunksize = 1000000
    for gm_chunk in pd.read_csv(path, chunksize=chunksize, sep=',', encoding='utf-8', usecols=columns, dtype=dtypes):
        resultTemp = pd.merge(dfSIRET['siret'], gm_chunk, on=['siret'], copy=False)
        result = pd.concat([result, resultTemp], axis=0, copy=False)
        del resultTemp
    result = result.drop_duplicates(subset=['siret'], keep='first')

    enrichissement_insee_siret = pd.merge(dfSIRET, result, how='outer', on=['siret'], copy=False)
    enrichissement_insee_siret.rename(columns={"siren_x": "siren"}, inplace=True)
    enrichissement_insee_siret.drop(columns=["siren_y"], axis=1, inplace=True)
    nanSiret = enrichissement_insee_siret[enrichissement_insee_siret.activitePrincipaleEtablissement.isnull()]
    enrichissement_insee_siret = enrichissement_insee_siret[
        enrichissement_insee_siret.activitePrincipaleEtablissement.notnull()]
    nanSiret = nanSiret.loc[:, ["siret", "siren", "denominationSociale"]]

    """
    # si le siret n'est pas trouvé, on peut essayer de matcher le siren. Je préfère désactiver
    # result2 is the result of the inner join between dfSIRET and stock-etablissement on SIREN
    result2 = pd.DataFrame(columns=columns)
    for gm_chunk in pd.read_csv(path, chunksize=chunksize, sep=',', encoding='utf-8', usecols=columns):
        gm_chunk['siren'] = gm_chunk['siren'].astype(str)
        resultTemp = pd.merge(nanSiret['siren'], gm_chunk, on=['siren'])
        result2 = pd.concat([result2, resultTemp], axis=0)
    result2 = result2.drop_duplicates(subset=['siren'], keep='first')

    enrichissement_insee_siren = pd.merge(nanSiret, result2, on='siren')
    enrichissement_insee_siren.rename(columns={ "siret_y": "siret"}, inplace=True)
    enrichissement_insee_siren.drop(columns=["siret_x"], axis=1, inplace=True)
    myList = list(enrichissement_insee_siret.columns)
    enrichissement_insee_siren.columns = myList
    """

    # Concat des deux resultats
    enrichissementInsee = enrichissement_insee_siret  # pd.concat([enrichissement_insee_siret, enrichissement_insee_siren])

    """
    ## create nanSiren dataframe
    temp_df = pd.merge(nanSiret, result2, indicator=True, how="outer", on='siren')
    nanSiren = temp_df[temp_df['activitePrincipaleEtablissement'].isnull()]
    nanSiren = nanSiren.iloc[:20, :3]
    #nanSiren = nanSiren.iloc[:, :3]
    nanSiren.reset_index(inplace=True, drop=True)
    """

    temp_df = pd.merge(nanSiret, result, indicator=True, how="outer", on='siren', copy=False)
    del result
    nanSiret = temp_df[temp_df['activitePrincipaleEtablissement'].isnull()]
    nanSiret = nanSiret.iloc[:, :3]
    # nanSiren = nanSiren.iloc[:, :3]
    nanSiret.reset_index(inplace=True, drop=True)

    return [enrichissementInsee, nanSiret]



def get_enrichissement_scrap(nanSiren, archiveErrorSIRET):
    # Enrichissement des données restantes

    # ....... Solution complémentaire pour ceux non-identifié dans la BDD
    columns = [
        'index',
        'rue',
        'siret',
        'ville',
        'typeEntreprise',
        'codeType',
        'detailsType',
        'SIRETisMatched']

    filter = 10
    nanSiren = nanSiren.iloc[:filter, :]

    df_scrap = pd.DataFrame(columns=columns)
    """
    for i in tqdm(range(len(nanSiren))):
        try:
            scrap = get_scrap_dataframe(i, nanSiren.siret[i])
            df_scrap = pd.concat([df_scrap, scrap], axis=0)
        except:
            try:
                scrap = get_scrap_dataframe(i, nanSiren.siren[i])
                df_scrap = pd.concat([df_scrap, scrap], axis=0)
            except:
                scrap = pd.DataFrame([i, ' ', ' ', ' ', ' ', ' ', ' ', False]).T
                scrap.columns = ['index', 'rue', 'siret', 'ville', 'typeEntreprise', 'codeType', 'detailsType',
                                 'SIRETisMatched']
                df_scrap = pd.concat([df_scrap, scrap], axis=0)
                pass
    """

    # Récupération des résultats
    nanSiren.reset_index(inplace=True)
    resultat = pd.merge(nanSiren, df_scrap, on='index', copy=False)
    resultatScrap1 = resultat[resultat.rue != ' ']

    # Données encore manquantes
    dfDS = resultat[resultat.rue == ' ']
    dfDS = dfDS.iloc[:, 1:4]
    dfDS.columns = ['siret', 'siren', 'denominationSociale']
    dfDS.reset_index(inplace=True, drop=True)

    """
    def requete(nom):
        pager.get('https://www.infogreffe.fr/recherche-siret-entreprise/chercher-siret-entreprise.html')
        pager.find_element_by_xpath('//*[@id="p1_deno"]').send_keys(nom, Keys.ENTER)
        time.sleep(2)
        url = pager.current_url
        return url

    options = Options()
    options.add_argument('--headless')
    pager = webdriver.Firefox(options=options)
    # pager = webdriver.PhantomJS('webdriver/phantomjs.exe')
    """

    df_scrap2 = pd.DataFrame(
        columns=['index', 'rue', 'siret', 'ville', 'typeEntreprise', 'codeType', 'detailsType', 'SIRETisMatched'])
    """
    for i in tqdm(range(len(dfDS))):
        try:
            url = requete(dfDS.denominationSociale[i])

            page = requests.get(url)
            tree = html.fromstring(page.content)

            rueSiret = tree.xpath('//div[@class="identTitreValeur"]/text()')
            infos = tree.xpath('//p/text()')
            details = tree.xpath('//a/text()')

            index = i
            rue = rueSiret[1]
            siret = rueSiret[5].replace(" ", "")
            ville = infos[7]
            typeEntreprise = infos[15]
            codeType = infos[16].replace(" : ", "")
            detailsType1 = details[28]
            detailsType2 = details[29]
            SIRETisMatched = (siret == dfDS.siret[i])
            if (detailsType1 == ' '):
                detailsType = detailsType2
            else:
                detailsType = detailsType1

            scrap2 = pd.DataFrame([index, rue, siret, ville, typeEntreprise, codeType, detailsType, SIRETisMatched]).T
            scrap2.columns = ['index', 'rue', 'siret', 'ville', 'typeEntreprise', 'codeType', 'detailsType',
                              'SIRETisMatched']
            df_scrap2 = pd.concat([df_scrap2, scrap2], axis=0)
        except:
            index = i
            scrap2 = pd.DataFrame([index, ' ', ' ', ' ', ' ', ' ', ' ', False]).T
            scrap2.columns = ['index', 'rue', 'siret', 'ville', 'typeEntreprise', 'codeType', 'detailsType',
                              'SIRETisMatched']
            df_scrap2 = pd.concat([df_scrap2, scrap2], axis=0)
            pass
    pager.quit()
    """

    # Récupération des résultats
    dfDS.reset_index(inplace=True)
    resultat = pd.merge(dfDS, df_scrap2, on='index', copy=False)
    resultatScrap2 = resultat[resultat.rue != ' ']

    ###############################################################################
    # Enregistrement des entreprises n'ayant aucune correspondance
    errorSIRET = resultat[
        (resultat.siret_y == '') | (resultat.siret_y == '') | (resultat.siret_y == ' ') | (resultat.siret_y.isnull())]
    errorSIRET = errorSIRET[['siret_x', 'siren', 'denominationSociale']]
    errorSIRET.columns = ['siret', 'siren', 'denominationSociale']
    errorSIRET.reset_index(inplace=True, drop=True)
    errorSIRET = pd.concat([errorSIRET, archiveErrorSIRET], axis=0, copy=False)
    errorSIRET = errorSIRET.drop_duplicates(subset=['siret', 'siren', 'denominationSociale'], keep='first')
    errorSIRET.to_csv('errorSIRET.csv', sep=';', index=False, header=True, encoding='utf-8')
    ###############################################################################

    # On réuni les résultats du scraping
    enrichissementScrap = pd.concat([resultatScrap1, resultatScrap2], copy=False)
    return enrichissementScrap



def enrichissement_acheteur(df):
    # Enrichissement des données des acheteurs #
    # Enrichissement des données via les codes siret/siren #
    # Utilisation d'un autre data frame pour traiter les Siret unique : acheteur.id

    with open('df_backup_cpv', 'rb') as df_backup_cpv:
        df = pickle.load(df_backup_cpv)

    dfAcheteurId = df['acheteur.id'].to_frame()
    dfAcheteurId.columns = ['siret']
    dfAcheteurId = dfAcheteurId.drop_duplicates(keep='first')
    dfAcheteurId.reset_index(inplace=True, drop=True)
    dfAcheteurId = dfAcheteurId.astype(str)

    # StockEtablissement_utf8
    chemin = os.path.join(path_to_data, conf["base_sirene_insee"])
    # chemin = 'dataEnrichissement/StockEtablissement_utf8.csv'
    result = pd.DataFrame(columns=['siret', 'codePostalEtablissement',
                                   'libelleCommuneEtablissement', 'codeCommuneEtablissement'])
    for gm_chunk in pd.read_csv(
            chemin, chunksize=1000000, sep=',', encoding='utf-8',
            usecols=['siret', 'codePostalEtablissement', 'libelleCommuneEtablissement', 'codeCommuneEtablissement']):
        gm_chunk['siret'] = gm_chunk['siret'].astype(str)
        resultTemp = pd.merge(dfAcheteurId, gm_chunk, on="siret", copy=False)
        result = pd.concat([result, resultTemp], axis=0, copy=False)
    result = result.drop_duplicates(subset=['siret'], keep='first')

    """
    dfAcheteurId["siren"] = np.nan
    dfAcheteurId.siren = dfAcheteurId.siret.str[:siren_len]
    chemin = 'dataEnrichissement/StockEtablissement_utf8.csv'
    result2 = pd.DataFrame(
        columns=['siren', 'codePostalEtablissement', 'libelleCommuneEtablissement', 'codeCommuneEtablissement'])
    for gm_chunk in pd.read_csv(chemin, chunksize=1000000, sep=',', encoding='utf-8',
                                usecols=['siren', 'codePostalEtablissement',
                                         'libelleCommuneEtablissement',
                                         'codeCommuneEtablissement']):
        gm_chunk['siren'] = gm_chunk['siren'].astype(str)
        resultTemp = pd.merge(dfAcheteurId, gm_chunk, on="siren")
        result2 = pd.concat([result2, resultTemp], axis=0)
    result2 = result2.drop_duplicates(subset=['siren'], keep='first')
    siret = pd.DataFrame(result['siret'])
    siret.columns = ['s']
    result2 = pd.merge(result2, siret, how='outer', left_on='siret', right_on='s')
    result2 = result2[result2.s.isnull()]

    dfManquant = pd.merge(dfAcheteurId, result, how='outer', on='siret')
    dfManquant = dfManquant[dfManquant['codeCommuneEtablissement'].isnull()]
    dfManquant = dfManquant.iloc[:, :2]
    result2 = pd.merge(dfManquant, result2, how='inner', on='siren')
    result2.columns = ['siret', 'codePostalEtablissement', 'libelleCommuneEtablissement', 'codeCommuneEtablissement']
    """

    enrichissementAcheteur = result
    enrichissementAcheteur.columns = ['acheteur.id', 'codePostalAcheteur', 'libelleCommuneAcheteur',
                                      'codeCommuneAcheteur']
    enrichissementAcheteur = enrichissementAcheteur.drop_duplicates(subset=['acheteur.id'], keep='first')

    df = pd.merge(df, enrichissementAcheteur, how='left', on='acheteur.id', copy=False)
    del enrichissementAcheteur
    with open('df_backup_acheteur', 'wb') as df_backup_acheteur:
        pickle.dump(df, df_backup_acheteur)

    return df


def segmentation(df):
    ###############################################################################
    # ########################### Segmentation de marché ##########################
    ###############################################################################
    # ... Créer une bdd par villes (acheteur/client)
    dfBIN = df[['type', 'nature', 'procedure', 'lieuExecutionTypeCode']]
    # Arrangement du code du lieu d'exécution
    dfBIN['lieuExecutionTypeCode'] = np.where(dfBIN['lieuExecutionTypeCode'] == 'CODE ARRONDISSEMENT',
                                              'CODE DEPARTEMENT', dfBIN['lieuExecutionTypeCode'])
    dfBIN['lieuExecutionTypeCode'] = np.where(
        (dfBIN['lieuExecutionTypeCode'] == 'CODE COMMUNE') | (dfBIN['lieuExecutionTypeCode'] == 'CODE POSTAL'),
        'CODE COMMUNE/POSTAL', dfBIN['lieuExecutionTypeCode'])

    # ... On binarise les variables qualitatives
    def binateur(data, to_bin):
        data = data.copy()
        X = data[to_bin]
        X = pd.get_dummies(X)
        data = data.drop(columns=to_bin)
        X = X.fillna(0)
        return pd.concat([data, X], axis=1, copy=False)

    dfBIN = binateur(dfBIN, dfBIN.columns)

    # ... Selection des variables quantitatives + nom de la commune
    dfNoBin = df[
        ['libelleCommuneAcheteur', 'montant', 'dureeMois', 'dureeMoisCalculee', 'distanceAcheteurEtablissement']]
    # Création d'une seule colonne pour la durée du marché
    dfNoBin['duree'] = round(dfNoBin.dureeMoisCalculee, 0)
    # On modifie les valeurs manquantes pour la distance en appliquant la médiane
    dfNoBin.distanceAcheteurEtablissement = np.where(dfNoBin['distanceAcheteurEtablissement'].isnull(),
                                                     dfNoBin['distanceAcheteurEtablissement'].median(),
                                                     dfNoBin['distanceAcheteurEtablissement'])

    # On obtient alors notre df prêt sans variables qualitatives (sauf libellé)
    df = dfNoBin.join(dfBIN)
    df = df[df['libelleCommuneAcheteur'].notnull()]
    df['nbContrats'] = 1  # Trouver autre solution
    df = df.groupby(['libelleCommuneAcheteur']).sum().reset_index()

    # ... Fréquence
    ensemble = ['type_Contrat de concession', 'type_Marché', 'nature_ACCORD-CADRE', 'nature_CONCESSION DE SERVICE',
                'nature_CONCESSION DE SERVICE PUBLIC', 'nature_CONCESSION DE TRAVAUX',
                'nature_DELEGATION DE SERVICE PUBLIC', 'nature_MARCHÉ',
                'nature_MARCHÉ DE PARTENARIAT', 'nature_MARCHÉ HORS ACCORD CADRE', 'nature_MARCHÉ SUBSÉQUENT',
                "procedure_Appel d'offres ouvert",
                "procedure_Appel d'offres restreint", 'procedure_Dialogue compétitif',
                'procedure_Marché négocié sans publicité ni mise en concurrence préalable',
                'procedure_Marché public négocié sans publicité ni mise en concurrence préalable',
                'procedure_Procédure adaptée', 'procedure_Procédure avec négociation',
                'procedure_Procédure non négociée ouverte', 'procedure_Procédure non négociée restreinte',
                'procedure_Procédure négociée ouverte',
                'procedure_Procédure négociée restreinte', 'lieuExecutionTypeCode_CODE CANTON',
                'lieuExecutionTypeCode_CODE COMMUNE/POSTAL',
                'lieuExecutionTypeCode_CODE DEPARTEMENT', 'lieuExecutionTypeCode_CODE PAYS',
                'lieuExecutionTypeCode_CODE REGION']
    for x in ensemble:
        df[x] = df[x] / df['nbContrats']

    # ... Duree, montant et distance moyenne par ville (par rapport au nb de contrats)
    df.distanceAcheteurEtablissement = round(df.distanceAcheteurEtablissement / df['nbContrats'], 0)
    df.duree = round(df.duree / df['nbContrats'], 0)
    df['montantMoyen'] = round(df.montant / df['nbContrats'], 0)

    # Renomme des colonnes
    df = df.rename(columns={
        'montant': 'montantTotal',
        'distanceAcheteurEtablissement': 'distanceMoyenne',
        'duree': 'dureeMoyenne',
        'type_Contrat de concession': 'nbContratDeConcession',
        'type_Marché': 'nbMarché'})

    # ... Mettre les valeurs sur une même unité de mesure
    df_nom = pd.DataFrame(df.libelleCommuneAcheteur)
    # Les deux variables du dessous ne sont jamais utilisées?#
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df)

    # ... On réassemble le df
    df = df_nom.join(df)
    return df



def CAH(df):
    #############################################################################
    # Application de l'algorithme de classification ascendante hiérarchique - CAH
    # Avec les données normalisée
    # Générer la matrice des liens
    Z = linkage(df, method='ward', metric='euclidean')
    # Dendrogramme
    plt.title('CAH avec matérialisation des X classes')
    dendrogram(Z, labels=df.index, orientation='left', color_threshold=65)
    plt.show()
    # Récupération des classes
    groupes_cah = pd.DataFrame(fcluster(Z, t=65, criterion='distance'), columns=['segmentation_CAH'])
    # Ajout au df
    df = df.join(groupes_cah)

    # On créé une 4e catégorie avec toutes les valeurs seules
    df.reset_index(inplace=True)
    a = pd.DataFrame(df.groupby('segmentation_CAH')['index'].nunique())
    a.reset_index(inplace=True)
    a.columns = ['cluster', 'nb']
    a = a.sort_values(by='nb', axis=0, ascending=False)
    a.reset_index(inplace=True, drop=True)
    a = a.drop([0, 1, 2])
    a = list(a.cluster)
    # On remplace
    df['segmentation_CAH'] = df['segmentation_CAH'].replace(a, 0)
    df.segmentation_CAH = df.segmentation_CAH.astype(int)

    # Changement / TRI des clusters
    cahORDER = pd.DataFrame(df.groupby('segmentation_CAH')[['montantTotal', 'segmentation_CAH']].mean())
    cahORDER = cahORDER.sort_values(by='montantTotal')
    cahORDER = cahORDER[cahORDER.segmentation_CAH != 0]
    l = ['0'] + list(cahORDER.segmentation_CAH.unique())
    k = [0, 1, 2, 3]
    listCorrespondance = {x: y for x, y in zip(k, l)}
    for word, initial in listCorrespondance.items():
        df['segmentation_CAH'] = np.where(df['segmentation_CAH'] == initial, word, df['segmentation_CAH'])

    # On ajoute au dataframe principal
    df = df[['libelleCommuneAcheteur', 'segmentation_CAH']]
    df_decp = pd.merge(df, df, how='left', on='libelleCommuneAcheteur', copy=False)
    df_decp.segmentation_CAH = np.where(df_decp.segmentation_CAH.isnull(), 0, df_decp.segmentation_CAH)
    df_decp.segmentation_CAH = df_decp.segmentation_CAH.astype(int)


def carte(df):
    ###############################################################################
    # ........ CARTE DES MARCHES PAR VILLE
    df_carte = df[['latitudeAcheteur', 'longitudeAcheteur', 'libelleCommuneAcheteur']]
    df_carte = df_carte[df_carte['latitudeAcheteur'] != 'nan']
    df_carte = df_carte[df_carte['longitudeAcheteur'] != 'nan']
    df_carte = df_carte.drop_duplicates(subset=['latitudeAcheteur', 'longitudeAcheteur'], keep='first')
    df_carte.reset_index(inplace=True, drop=True)

    dfMT = df.groupby(['latitudeAcheteur', 'longitudeAcheteur']).montant.sum().to_frame(
        'montantTotal').reset_index()
    dfMM = df.groupby(['latitudeAcheteur', 'longitudeAcheteur']).montant.mean().to_frame(
        'montantMoyen').reset_index()
    dfIN = df.groupby(['latitudeAcheteur', 'longitudeAcheteur']).identifiantMarche.nunique().to_frame(
        'nbMarches').reset_index()
    dfSN = df.groupby(['latitudeAcheteur', 'longitudeAcheteur']).siretEtablissement.nunique().to_frame(
        'nbEntreprises').reset_index()
    dfDM = df.groupby(['latitudeAcheteur', 'longitudeAcheteur']).distanceAcheteurEtablissement.median().to_frame(
        'distanceMediane').reset_index()

    df_carte = pd.merge(df_carte, dfMT, how='left', on=['latitudeAcheteur', 'longitudeAcheteur'], copy=False)
    df_carte = pd.merge(df_carte, dfMM, how='left', on=['latitudeAcheteur', 'longitudeAcheteur'], copy=False)
    df_carte = pd.merge(df_carte, dfIN, how='left', on=['latitudeAcheteur', 'longitudeAcheteur'], copy=False)
    df_carte = pd.merge(df_carte, dfSN, how='left', on=['latitudeAcheteur', 'longitudeAcheteur'], copy=False)
    df_carte = pd.merge(df_carte, dfDM, how='left', on=['latitudeAcheteur', 'longitudeAcheteur'], copy=False)
    df_carte = pd.merge(df_carte, df, how='left', on=['libelleCommuneAcheteur'], copy=False)

    df_carte.montantTotal = round(df_carte.montantTotal, 0)
    df_carte.montantMoyen = round(df_carte.montantMoyen, 0)
    df_carte.nbMarches = round(df_carte.nbMarches, 0)
    df_carte.nbEntreprises = round(df_carte.nbEntreprises, 0)
    df_carte.distanceMediane = round(df_carte.distanceMediane, 0)
    df_carte.distanceMediane = np.where(df_carte.distanceMediane.isnull(), 0, df_carte.distanceMediane)

    ###############################################################################
    # Carte des DECP
    geojson = json.loads(urllib.request.urlopen('https://france-geojson.gregoiredavid.fr/repo/regions.geojson').read())
    df_Reg = df.groupby(['codeRegionAcheteur']).montant.sum().to_frame('montantMoyen').reset_index()
    df_Reg.columns = ['code', 'montant']
    df_Reg = df_Reg[(df_Reg.code != 'nan') & (df_Reg.code != '98')]
    df_Reg.montant = round(df_Reg.montant / 1000000, 0).astype(int)
    df_Reg.montant = np.where(df_Reg.montant > 10000, 10000, df_Reg.montant)

    path = os.path.join(path_to_data, conf["departements-francais"])
    depPop = pd.read_csv(path, sep='\t', encoding='utf-8',
                         usecols=['NUMÉRO', 'POPULATION'])
    depPop.columns = ['code', 'population']
    depPop.code = depPop.code.astype(str)
    depPop = depPop[depPop.population.notnull()]
    depPop.population = depPop.population.astype(int)
    for i in range(len(depPop)):
        if len(depPop['code'][i]) < 2:
            depPop['code'][i] = '0' + depPop['code'][i]

    geojson2 = json.loads(urllib.request.urlopen(
        'https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements-avec-outre-mer.geojson').read())
    df_Dep = df.groupby(['codeDepartementAcheteur']).montant.sum().to_frame('montantMoyen').reset_index()
    df_Dep.columns = ['code', 'montant']
    df_Dep = df_Dep[(df_Dep.code != 'nan')]
    df_Dep = pd.merge(df_Dep, depPop, how='left', on='code', copy=False)
    df_Dep = df_Dep[df_Dep.population.notnull()]
    df_Dep.montant = round(df_Dep.montant / df_Dep.population, 0).astype(int)
    df_Dep.montant = np.where(df_Dep.montant > 2000, 2000, df_Dep.montant)

    dfHM = df[['latitudeAcheteur', 'longitudeAcheteur']]
    dfHM = dfHM[(dfHM.latitudeAcheteur != 'nan') | (dfHM.longitudeAcheteur != 'nan')]

    # Mise en forme
    c = folium.Map(location=[47, 2.0], zoom_start=6, control_scale=True)
    plugins.MiniMap(toggle_display=True).add_to(c)

    mapMarker = folium.Marker([44, -4], icon=folium.features.CustomIcon(
        'https://upload.wikimedia.org/wikipedia/commons/thumb/3/35/Information_icon.svg/1000px-Information_icon.svg.png',
        icon_size=(20, 20)),
        popup=folium.Popup('<b>Indicateur de distance</b></br></br>'
                           + '<img src="https://icon-library.com/images/map-pin-icon/map-pin-icon-17.jpg"  width=8 height=8/>' + ' '
                           + '<img src="https://icon-library.com/images/map-pin-icon/map-pin-icon-17.jpg"  width=14 height=14/>' + ' '
                           + '<img src="https://icon-library.com/images/map-pin-icon/map-pin-icon-17.jpg"  width=20 height=20/> : Distance moyenne</br>entre acheteurs et entreprises' + '</br></br>'
                           + '<b>Ségmentation des acheteurs </b></br></br>'
                           + '<img src="https://icon-library.com/images/map-pin-icon/map-pin-icon-17.jpg"  width=20 height=20/> : Petit' + '</br>'
                           + '<img src="https://cdn1.iconfinder.com/data/icons/vibrancie-map/30/map_001-location-pin-marker-place-512.png"  width=20 height=20/> : Moyen' + '</br>'
                           + '<img src="https://cdn.cnt-tech.io/api/v1/tenants/dd1f88aa-e3e2-450c-9fa9-a03ea59a6bf0/domains/57a9d53a-fe30-4b6f-a4de-d624bd25134b/buckets/8f139e2f-9e74-4be3-9d30-d8f180f02fbb/statics/56/56d48498-d2bf-45f8-846e-6c9869919ced"  width=20 height=20/> : Grand' + '</br>'
                           + '<img src="https://svgsilh.com/svg/157354.svg"  width=20 height=20/> : Hors-segmentation',
                           max_width=320, show=True), overlay=False).add_to(c)

    marker_cluster = MarkerCluster(name='DECP par communes').add_to(c)
    for i in range(len(df_carte)):
        if (df_carte.segmentation_CAH[i] == 1):
            icon = folium.features.CustomIcon('https://icon-library.com/images/map-pin-icon/map-pin-icon-17.jpg',
                                              icon_size=(max(20, min(40, df_carte.distanceMediane[i] / 2)),
                                                         max(20, min(40, df_carte.distanceMediane[i] / 2))))
        elif (df_carte.segmentation_CAH[i] == 2):
            icon = folium.features.CustomIcon(
                'https://cdn1.iconfinder.com/data/icons/vibrancie-map/30/map_001-location-pin-marker-place-512.png',
                icon_size=(
                    max(20, min(40, df_carte.distanceMediane[i] / 2)),
                    max(20, min(40, df_carte.distanceMediane[i] / 2))))
        elif (df_carte.segmentation_CAH[i] == 3):
            icon = folium.features.CustomIcon(
                'https://cdn.cnt-tech.io/api/v1/tenants/dd1f88aa-e3e2-450c-9fa9-a03ea59a6bf0/domains/57a9d53a-fe30-4b6f-a4de-d624bd25134b/buckets/8f139e2f-9e74-4be3-9d30-d8f180f02fbb/statics/56/56d48498-d2bf-45f8-846e-6c9869919ced',
                icon_size=(
                    max(20, min(40, df_carte.distanceMediane[i] / 2)),
                    max(20, min(40, df_carte.distanceMediane[i] / 2))))
        else:
            icon = folium.features.CustomIcon('https://svgsilh.com/svg/157354.svg', icon_size=(
                max(20, min(40, df_carte.distanceMediane[i] / 2)), max(20, min(40, df_carte.distanceMediane[i] / 2))))

        folium.Marker([df_carte.latitudeAcheteur[i], df_carte.longitudeAcheteur[i]],
                      icon=icon,
                      popup=folium.Popup('<b><center>' + df_carte.libelleCommuneAcheteur[i] + '</center></b></br>'
                                         + '<b>' + df_carte.nbMarches[i].astype(str) + '</b> marchés '
                                         + 'pour un montant moyen de <b>' + round(df_carte.montantMoyen[i] / 1000,
                                                                                  0).astype(int).astype(str) + ' mille euros</b> '
                                         + "</br>avec <b>" + df_carte.nbEntreprises[i].astype(str) + ' entreprises</b> '
                                         + "à une distance médiane de <b>" + df_carte.distanceMediane[i].astype(int).astype(str) + ' km</b> ',
                                         max_width=320, min_width=200),
                      tooltip=folium.Tooltip(df_carte.libelleCommuneAcheteur[i]), clustered_marker=True).add_to(
            marker_cluster)

    HeatMap(data=dfHM[['latitudeAcheteur', 'longitudeAcheteur']], radius=10, name="HeatMap des marchés", show=False,
            overlay=False).add_to(c)

    choropleth = folium.Choropleth(geo_data=geojson, name='Régions', data=df_Reg, columns=['code', 'montant'],
                                   key_on='feature.properties.code', fill_color='YlGnBu', fill_opacity=0.7,
                                   line_opacity=0.2, nan_fill_color='#8c8c8c',
                                   highlight=True, line_color='black', show=False, overlay=False,
                                   legend_name="Montant total des marchés (en millions €)").add_to(c)
    choropleth.geojson.add_child(folium.features.GeoJsonTooltip(['nom'], labels=False))

    choropleth = folium.Choropleth(geo_data=geojson2, name='Départements', data=df_Dep, columns=['code', 'montant'],
                                   key_on='feature.properties.code', fill_color='YlOrRd', fill_opacity=0.7,
                                   line_opacity=0.2, nan_fill_color='#8c8c8c',
                                   highlight=False, line_color='black', show=False, overlay=False,
                                   legend_name="Montant total par habitants (en milliers €)").add_to(c)
    choropleth.geojson.add_child(folium.features.GeoJsonTooltip(['nom'], labels=False))

    folium.TileLayer('OpenStreetMap', overlay=True, show=True, control=False).add_to(c)
    folium.LayerControl(collapsed=False).add_to(c)
    c.save('carte/carteDECP.html')
