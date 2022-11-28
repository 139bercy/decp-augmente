import git 
import os
import boto3
import json
from saagieapi import SaagieApi


# Global variables.
local_credentials="saagie_cred.json"
local_credentials_exist = os.path.exists(local_credentials)
if local_credentials_exist : # Dans le cas où on fait tourner ça en local
    with open(local_credentials, "r") as f:
        credentials = json.load(f)
    ACCESS_KEY = credentials["ACCESS_KEY"]
    SECRET_KEY = credentials["SECRET_KEY"]
    USER =credentials["USER_SAAGIE"]
    PASSWORD = credentials["PASSWORD_SAAGIE"]
else :  # Sur la CI ou Saagie
    ACCESS_KEY = os.environ.get("ACCESS_KEY")
    SECRET_KEY = os.environ.get("SECRET_KEY")
    USER =os.environ.get("USER_SAAGIE")
    PASSWORD = os.environ.get("PASSWORD_SAAGIE")
PROJECT_NAME = "BercyHub - OpenData"
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
    bucket_name = "bercy"
    file_id_commit = "id_commit.txt"
    s3 = boto3.resource('s3', 
                        aws_access_key_id=ACCESS_KEY, 
                        aws_secret_access_key=SECRET_KEY, 
                        region_name="eu-west-3"
                        )
    object = s3.Object(bucket_name, file_id_commit)
    id_commit = object.get()['Body'].read().decode('ascii') # Le decode est nécessaire pour passer d'un bytes à un string.
    print(f"Le dernier commit à jour sur Saagie est le {id_commit}") 
    changed_files = REPO.git.diff(f"{id_commit}..HEAD", name_only=True) # On récupère uniquement les noms des fichiers qui ont été concernés par des commits depuis le dernier commit mis sur Saagie
    return changed_files.split('\n'), object
    
def updates_files_on_saagie(modified_files : list, object):
    """
    Cette fonction met à jour les jobs Saagie. Une fois que les jobs ont été mis à jour ou créés, on stock dans le fichier S3 correspondant l'ID du commit.
    
    Arguments
    modified_files (list) List containing the files who needs to be updated on Saagie
    object : the object used to write on s3 file.

    return 
    True si la fonction s'éxecute en entier.

    """   
    saagieapi =  SaagieApi.easy_connect(url_saagie_platform="https://saagieavv-workspace.saagie.com/projects/platform/1/project/3581976c-20f6-46e1-892b-2fa168c7159b/jobs",
                   user=USER,
                   password=PASSWORD)

    for file in modified_files:
        file_name = file[:-3]
        print(f"Traitement du fichier {file_name}")
        try:
            id_job = saagieapi.jobs.get_id(project_name=PROJECT_NAME, job_name=str(file_name))
            print(f"Un jobs {file_name} a été trouvé, son id est le {id_job}. On le met à jour.")
            saagieapi.jobs.upgrade(job_id=id_job, file=file, command_line="python {file}")
        except :
            print(f"Il n'existe pas de jobs {file_name}. On le créé avec les paramètres par défaut.")
            id_projet = saagieapi.projects.get_id(PROJECT_NAME)
            saagieapi.jobs.create(job_name=str(file_name), file=file, command_line="python {file}", project_id=id_projet,
            category='Extraction',
            technology='python',# technology id corresponding to your context.id in your technology catalog definition
            technology_catalog='Saagie',
            runtime_version='3.9')
    print('Actualisation du dernier commit traité')
    id_commit = REPO.head.commit
    object.put(Body=str(id_commit))
    return True

def main():
    modified_files, object = get_files_to_updates()    
    bool = updates_files_on_saagie(modified_files, object)
    return None


if __name__ == "__main__":
    main()