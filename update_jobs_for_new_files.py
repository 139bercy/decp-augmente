import git 
import os
import boto3
import json
from zipfile import ZipFile
import botocore
from saagieapi import SaagieApi


# Global variables.
local_credentials="saagie_cred.json"
local_credentials_exist = os.path.exists(local_credentials)
if local_credentials_exist :  # Dans le cas où on fait tourner ça en local
    with open(local_credentials, "r") as f:
        credentials = json.load(f)
    ACCESS_KEY = credentials["ACCESS_KEY"]
    SECRET_KEY = credentials["SECRET_KEY"]
    USER =credentials["USER_SAAGIE"]
    PASSWORD = credentials["PASSWORD_SAAGIE"]
    ENDPOINT_S3 = credentials["ENDPOINT_S3"]
    PROJECT_NAME = credentials["PROJECT_NAME"]
    BUCKET_NAME = credentials["BUCKET_NAME"]
else :  # Sur la CI ou Saagie
    ACCESS_KEY = os.environ.get("ACCESS_KEY")
    SECRET_KEY = os.environ.get("SECRET_KEY")
    USER =os.environ.get("USER_SAAGIE")
    PASSWORD = os.environ.get("PASSWORD_SAAGIE")
    ENDPOINT_S3 = os.environ.get("ENDPOINT_S3")
    PROJECT_NAME = os.environ.get("PROJECT_NAME")
    BUCKET_NAME = os.environ.get("BUCKET_NAME")

dir_path = os.path.dirname(os.path.realpath(__file__))
REPO = git.Repo(dir_path)


def get_files_to_updates():
    """
    Cette fonction récupère le dernier id du commit pour lequel les fichiers ont été updatés sur Saagie.

    Returns
    -------
    changed_files (list) List containing the files who needs to be updated on Saagie
    object : the object used to write on s3 file.
    """
    # Connexion
    bucket_name = BUCKET_NAME
    file_id_commit = "id_commit.txt"
    s3 = boto3.resource(service_name = 's3', 
                        aws_access_key_id=ACCESS_KEY,
                        aws_secret_access_key=SECRET_KEY,
                        region_name="gra",
                        endpoint_url="https://"+str(ENDPOINT_S3))
    # On regarde si l'object id_commit existe
    try:
        s3.Object(bucket_name, file_id_commit).load()
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("Le fichier recherché n'existe pas. Pensez à l'initialiser.")
    object = s3.Object(bucket_name, file_id_commit)
    id_commit = object.get()['Body'].read().decode('ascii') # Le decode est nécessaire pour passer d'un bytes à un string.
    print(f"Le dernier commit à jour sur Saagie est le {id_commit}") 
    changed_files = REPO.git.diff(f"{id_commit}..HEAD", name_only=True) # On récupère uniquement les noms des fichiers qui ont été concernés par des commits depuis le dernier commit mis sur Saagie
    return changed_files.split('\n'), object


def updates_files_on_saagie(modified_files : list,
                            object,
                            files_to_zip_with_utils=["gestion_flux.py", "nettoyage.py", "enrichissement.py", "upload_dataeco.py"]):
    """
    Cette fonction met à jour les jobs Saagie. Une fois que les jobs ont été mis à jour ou créés, on stock dans le fichier S3 correspondant l'ID du commit.
    
    Arguments
    modified_files (list) List containing the files who needs to be updated on Saagie
    object : the object used to write on s3 file.
    files_to_zip_with_utils (list) : Somes files need a script utils.py to be fully functional. We have to zip them to upgrade a job cleanly.

    return 
    result (dict) Indique l'état de la communication avec le S3 pour l'actualisation du commit id.

    """   
    saagieapi =  SaagieApi.easy_connect(url_saagie_platform="https://mefsin-workspace.pcv.saagie.io/projects/platform/1/project/4fbca8d8-b3a5-4f63-97f1-b2ca6362a2b2/jobs",
                                        user=USER,
                                        password=PASSWORD)

    for file in modified_files:
        file_name = file[:-3]
        if len(file_name) != 0: 
            print(f"Traitement du fichier {file_name}")
            try:
                id_job = saagieapi.jobs.get_id(project_name=PROJECT_NAME, job_name=str(file_name))
                print(f"Un jobs {file_name} a été trouvé, son id est le {id_job}. On le met à jour.")
                # Par sécurité, tous les jobs seront upgradés avec le fichier de requirements correspondant au job.
                zipObj = ZipFile(f"{file_name}.zip", "w")
                zipObj.write(file)
                zipObj.write("requirements.txt")
                if file in files_to_zip_with_utils:
                    zipObj.write('utils.py')
                zipObj.close()

                if file_name == "upload_dataeco":

                    saagieapi.jobs.upgrade(job_id=id_job, file=f"{file_name}.zip", command_line=f"apt-get update \napt-get install lftp \npython {file_name}.py")
                else:
                    saagieapi.jobs.upgrade(job_id=id_job, file=f"{file_name}.zip", command_line=f"python {file_name}.py")
            except :
                print(f"Il n'existe pas de jobs {file_name}. On le créé avec les paramètres par défaut.")
                id_projet = saagieapi.projects.get_id(PROJECT_NAME)
                zipObj = ZipFile(f"{file_name}.zip", "w")
                zipObj.write(file)
                zipObj.write("requirements.txt")
                if file in files_to_zip_with_utils:
                    zipObj.write('utils.py')
                zipObj.close()
                saagieapi.jobs.create(job_name=str(file_name), file=f"{file_name}.zip", command_line=f"python {file_name}.py", project_id=id_projet,
                category='Extraction',
                technology='python',  # technology id corresponding to your context.id in your technology catalog definition
                technology_catalog='Saagie',
                runtime_version='3.9')
    print('Actualisation du dernier commit traité')
    id_commit = REPO.head.commit
    result = object.put(Body=str(id_commit))
    return result


def main():
    modified_files, object = get_files_to_updates()    
    updates_files_on_saagie(modified_files, object)
    return None


if __name__ == "__main__":
    main()