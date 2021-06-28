import os, uuid
from azure.storage.blob import (
    BlobServiceClient,
    BlobClient,
    ContainerClient,
    __version__,
)
import shutil
import traceback
import time
import logging

import sys

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from settings import env

MODEL_PATH = os.path.join(BASE_PATH, "data")
logging.info("Downloading data from blob")
tic = time.time()
try:
    try:
        os.makedirs(MODEL_PATH)
        logging.info("created data folder")
    except Exception as ex:
        logging.error("data folder exists")
    local_path = MODEL_PATH
    file_names = [
        "keywords_list_md.npy",
        "relations_dict_md.npy",
        "wiki_data_relation_embeddings_md.npy",
        "keywords_list_ft.npy",
        "relations_dict_ft.npy",
        "wiki_data_relation_embeddings_ft.npy",
    ]
    connect_str = env.AZURE_STORAGE_CONNECTION_STRING
    # Create the BlobServiceClient object which will be used to create a container client
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    # Instantiate a ContainerClient
    container_client = blob_service_client.get_container_client("lmkgfiles")
    for f in file_names:
        logging.info(f"downloadin file {f}")
        download_file_path = os.path.join(local_path, f)
        blob_client = blob_service_client.get_blob_client(container="lmkgfiles", blob=f)
        blob_list = container_client.list_blobs()
        for blob in blob_list:
            if blob.name == f:
                with open(download_file_path, "wb") as download_file:
                    download_file.write(blob_client.download_blob().readall())
                break

except Exception as ex:
    logging.error(ex)
    traceback.print_exc()

toc = time.time()
time_taken = toc - tic
logging.info(
    f"Time taken to download the files from blob (s): {time_taken}",
)

logging.info("Unzipping files: ")
try:
    for f in os.listdir(MODEL_PATH):
        if f.endswith(".zip"):
            logging.info(f"unzipping file {f}")
            f_folder = f.replace(".zip", "")
            logging.info(f"Extraction complete for file {f}")
            shutil.unpack_archive(os.path.join(MODEL_PATH, f), MODEL_PATH)
            logging.info("Deleting file: {f}")
            os.remove(os.path.join(MODEL_PATH, f))
except Exception as ex:
    logging.error(ex)
    traceback.print_exc()
