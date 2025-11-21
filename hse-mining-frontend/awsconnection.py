import awswrangler as wr
import pandas as pd
import boto3
import os
from dotenv import load_dotenv

load_dotenv()

def load_data(query: str, params: dict = None) -> pd.DataFrame:
    DATABASE = os.getenv('DATABASE')
    WORKGROUP = os.getenv('WORKGROUP')
    S3_OUTPUT = os.getenv('S3_OUTPUT')
    DATA_SOURCE = os.getenv('DATA_SOURCE')
    ATHENA_ACCESS_KEY = os.getenv('ATHENA_ACCESS_KEY')
    ATHENA_SECRET_ACCESS_KEY = os.getenv('ATHENA_SECRET_ACCESS_KEY')

    session = None
    
    if ATHENA_ACCESS_KEY and ATHENA_SECRET_ACCESS_KEY:
        session = boto3.Session(
            aws_access_key_id=ATHENA_ACCESS_KEY,
            aws_secret_access_key=ATHENA_SECRET_ACCESS_KEY
        )

    observations = wr.athena.read_sql_query(
        query,
        database=DATABASE,
        s3_output=S3_OUTPUT,
        paramstyle="named",
        params=params,
        workgroup=WORKGROUP,
        data_source=DATA_SOURCE,
        ctas_approach=False,
        boto3_session=session
    )

    observations.columns = list(map(str.lower, observations.columns))
    return observations