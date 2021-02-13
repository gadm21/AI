
import zipfile
import os
import json

import boto3
from botocore.exceptions import ClientError


bucket_name = 'randomfirstbucket'
train_prefix = 'Input/train/'
val_prefix = 'Input/val/'
train_bucket_prefix = bucket_name + '/' + train_prefix 
val_bucket_prefix = bucket_name + '/' + val_prefix 

zipped_data = r'C:\Users\gad\Desktop\repos\AI\aws\data\dataset.zip'
input_data_dir = r'C:\Users\gad\Desktop\repos\AI\aws\data\input'
train_dir = os.path.join(input_data_dir, 'train')
small_train_dir = os.path.join(input_data_dir, 'small_train')
test_dir = os.path.join(input_data_dir, 'test')
annotated_data_dir = r'C:\Users\gad\Desktop\repos\AI\aws\data\annotated'
manifest_file = r'C:\Users\gad\Desktop\repos\AI\aws\data\output.manifest'


def extract_files(src_file, dst_dir):
    with zipfile.ZipFile(src_file, 'r') as file :
        file.extractall(dst_dir)


def read_manifest(file):
    with open(file, 'r') as f :
        return f.read()




def create_bucket( bucket_name):

    s3_resource = boto3.resource('s3')
    response = s3_resource.create_bucket(Bucket = bucket_name)

    return response

def upload_file( bucket_name, file_name, dst_folder = None):
    s3_resource = boto3.resource('s3')
    
    obj = s3_resource.Bucket(bucket_name).Object('Input')
    status = obj.upload_file(Filename = file_name)
    return status

def create_dir(bucket_name, dir_name):
    if dir_name[-1] != '/' :
        dir_name += '/'
    s3_resource = boto3.resource('s3')
    bucket = s3_resource.Bucket(bucket_name)
    status = bucket.new_key(dir_name)
    return status

def get_contents(bucket):
    s3 = boto3.client('s3')
    bucket = s3.list_objects(Bucket = bucket_name)
    contents = [file['Key'] for file in bucket['Contents']]
    return contents