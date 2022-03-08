import os
import pyodbc
from datetime import datetime
from azure.cosmosdb.table.models import Entity
import pandas as pd
from pandas.compat import StringIO #if this doesn't work try: from io import StringIO
import time


import azure.functions as func

from azure.cosmosdb.table.tableservice import TableService

import logging
logging.basicConfig(filename = 'log.log', level = logging.INFO)

# table storage keys
storage_key = os.environ["faunadata"]
table_service = TableService(connection_string=storage_key)
# sql key
sql_conn = os.environ["faunadb"]


def create_table(sessionid, table_service):
    try:
        table_name = 'Features' + str(sessionid)
        if not table_service.exists(table_name):
            table_service.create_table(table_name)            
        return table_name
    except:
        logging.error("Error in creating Table Storage")


def ConvertDateTime(date_id, time_id):
    try:
        TimeLength = len(time_id)
        if(TimeLength == 6):
            hour = int(time_id[0:2])
            minute = int(time_id[2:4])
            second = int(time_id[4:6])
        elif(TimeLength == 5):
            hour = int(time_id[0:1])
            minute = int(time_id[1:3])
            second = int(time_id[3:5])
        elif(TimeLength == 4):
            hour = 00
            minute = int(time_id[0:2])
            second = int(time_id[2:4])
        elif(TimeLength == 3):
            hour = 00
            minute = int(time_id[0:1])
            second = int(time_id[1:3])
        CalcDateTime = datetime(year=int(date_id[0:4]), month=int(
            date_id[4:6]), day=int(date_id[6:8]), hour=hour, minute=minute, second=second)
        return CalcDateTime
    except:
        logging.error("Error in ConvertDateTime")

def get_feature(sql_conn):
    conn = pyodbc.connect(sql_conn)
    cursor = conn.cursor()
    sql_query = pd.read_sql("Select Name, Id from Feature", conn)
    return sql_query

def insert_features_data(features, table_name, table_service, sql_conn):
        
    features_indexed = get_feature(sql_conn)
    indexed_dict = dict(zip(features_indexed['Name'], features_indexed['Id']))

    try:
        for index, row in features.iterrows(): 
            FeatureId = indexed_dict[row['Name']]
            CalcDateTime = ConvertDateTime(row['Calcdateid'], row['Calctimeid'])
            MeasDateTime = ConvertDateTime(row['Measdateid'], row['Meastimeid'])
            feat = insert_features(SegmentId=row['Segment'], Wavelength=row['Wavelength'], Value=row['Value'], Measdateid=row['Measdateid'], Meastimeid=row['Meastimeid'], MeasurementCode=row['MeasurementCode'],
            Calcdateid=row['Calcdateid'], Calctimeid=row['Calctimeid'], SessionId=row['SessionId'], FeatureId=FeatureId, CalcDateTime=CalcDateTime, 
                                   MeasDateTime=MeasDateTime)
            table_service.insert_or_replace_entity(table_name, feat, timeout=None)
        logging.info("Data inserted successfully to table storage")
    except:
        logging.error("Error in inserting features data")

def insert_features(SegmentId, Wavelength, Value, Measdateid, Meastimeid, MeasurementCode, Calcdateid, Calctimeid, SessionId, FeatureId, CalcDateTime, MeasDateTime):
    
    WavelengthId = [str(Wavelength), '000'][Wavelength == 'N/A']

    rowkey_value = str(Measdateid).zfill(5) + '-' + str(Meastimeid).zfill(5) + '-' + \
        str(MeasurementCode) + '-' + \
        str(SegmentId).zfill(5) + '-' + WavelengthId
    feature = Entity()
    feature.PartitionKey = str(FeatureId).zfill(16)
    feature.RowKey = str(rowkey_value)
    feature.FeatureId = FeatureId
    feature.SegmentId = SegmentId
    feature.WavelengthId = Wavelength
    feature.Value = Value
    feature.MeasDateId = Measdateid
    feature.MeasTimeId = Meastimeid
    feature.MeasurementCode = MeasurementCode
    feature.DateId = Calcdateid
    feature.TimeId = Calctimeid
    feature.SessionID = SessionId
    feature.CalcDateTime = CalcDateTime
    feature.MeasDateTime = MeasDateTime
    return feature


def main(message: func.ServiceBusMessage):

    body = message.get_body().decode('utf-8')
    features_str = body['features']
    features = pd.read_csv(StringIO(features_str), sep='\s+')
    sessionid = body['sessionid']

    # save to table storage
    table_name = create_table(sessionid, table_service)
    start = time.time()
    insert_features_data(features, table_name, table_service, sql_conn)
    duration = time.time() - start
    logging.warn(f"{duration} for insert")
