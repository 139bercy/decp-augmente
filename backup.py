import utils
import datetime

source = {
    'Bucket' : utils.BUCKET_NAME,
    'Key' : 'data/cache_df'
}
today = datetime.date.today()
backup_name = "save/data/cache_df_"+today.strftime("%Y-%m-%d")
utils.s3.meta.client.copy(source, utils.BUCKET_NAME, backup_name)