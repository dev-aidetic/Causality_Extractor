from flask import Flask, request, Response
from flask_cors import CORS
import json
import traceback
import time
import os
import numpy as np
import time
import requests
import spacy
import nltk


################################ Loading Models ##############################
nlp = spacy.load("en_core_web_md")

############################### IMPORT CUSTOM PACKAGES ############################
from rule_based_extractor.extract import GetCausalReln

# from lmkg.utils.utils import intialize_logging
# from lmkg.handler import triplet_handler
from utils import rearrange_result

############################### SECRET ID CHECK ############################


def check_for_secret_id(request_data):

    try:
        if "secret_id" not in request_data.keys():
            return False, "Secret Key Not Found."

        else:
            if request_data["secret_id"] == "aidetic":
                return True, "Secret Key Matched"
            else:
                return False, "Secret Key Does Not Match. Incorrect Key."
    except Exception as e:
        message = "Error while checking secret id: " + str(e)
        return False, message


app = Flask(__name__)
CORS(app)

###########################################################################
# CAUSAL RELATIONSHIP EXTRACTOR
###########################################################################


@app.route("/causal_relatonship_extractor", methods=["POST"])
def causal_relatonship():

    start = time.time()
    try:

        # GET JSON DATA
        request_data = request.get_json()
        print("Request Data: ", request_data)

        # CHECK FOR SECRET ID
        secret_id_status, secret_id_message = check_for_secret_id(request_data)
        print("Secret ID Check: ", secret_id_status, secret_id_message)
        if not secret_id_status:
            data = {}
            data["message"] = secret_id_message
            data["success"] = False
            if "request_id" in request_data.keys():
                data["request_id"] = request_data["request_id"]
            else:
                data["request_id"] = None
            data = json.dumps(data)
            resp = Response(data, status=401, mimetype="application/json")

            end = time.time()
            print(f"Time taken {end-start:.3f} secs")
            return resp

        # PROCESS TEXT AND RETURN RESPONSE
        if "data" in request_data.keys():
            working_data = request_data["data"]
            data = {}
            temp = []
            for wd in working_data:
                if "text" not in wd.keys():
                    continue
                else:
                    np_link = True
                    if "np_link" in wd.keys():
                        if wd["np_link"] == False:
                            np_link = False

                    lmkg = True
                    if "lmkg" in wd.keys():
                        if wd["lmkg"] == False:
                            lmkg = False

                    rule_based = True
                    if "rule_based" in wd.keys():
                        if wd["rule_based"] == False:
                            rule_based = False
                    text = wd["text"]
                    lmkg_results = []

                    rule_based_result = []
                    if lmkg:
                        lmkg_results = {}
                        lmkg_results["text"] = text
                        lmkg_results["relations"] = []
                        # lmkg_results = triplet_handler.relationship_handler(text)
                        sent_text = nltk.sent_tokenize(text)
                        for sent in sent_text:
                            api_request = {"text": sent}

                            lmkg_results_ = requests.post(
                                "http://164.52.201.205:8000/lmkg_triplets",
                                json=api_request,
                            ).json()
                            lmkg_results["relations"].extend(lmkg_results_["relations"])

                    if rule_based:
                        CausalReln = GetCausalReln(
                            nlp,
                            text,
                            np_link=np_link,
                            load_from_file=False,
                        )
                        rule_based_result = CausalReln.get_causal_relations()

                    temp_dict = {}
                    temp_dict["processed_data"] = {}
                    (
                        temp_dict["processed_data"][
                            "attention_based_bert_causality_triplets"
                        ],
                        temp_dict["processed_data"][
                            "dynamic_dependency_cardinality_mapper"
                        ],
                    ) = rearrange_result(lmkg_results, rule_based_result)

                temp.append(temp_dict)

            data["complete_processed_data"] = temp
            data["success"] = True
            if "request_id" in request_data.keys():
                data["request_id"] = request_data["request_id"]
            else:
                data["request_id"] = None
            data = json.dumps(data)
            resp = Response(data, status=200, mimetype="application/json")
            end = time.time()
            print(f"Time taken {end-start:.3f} secs")
            return resp
        else:
            data = {}
            data["message"] = "Invalid Request. Data Not Found For Processing."
            data["success"] = False
            if "request_id" in request_data.keys():
                data["request_id"] = request_data["request_id"]
            else:
                data["request_id"] = None
            data = json.dumps(data)
            resp = Response(data, status=400, mimetype="application/json")
            end = time.time()
            print(f"Time taken {end-start:.3f} secs")
            return resp

    except Exception as e:
        traceback.print_exc()
        data = {}
        data["message"] = "Error While Processing request: " + str(e)
        data["success"] = False
        try:
            if "request_id" in request_data.keys():
                data["request_id"] = request_data["request_id"]
            else:
                data["request_id"] = None
        except Exception as e:
            data["request_id"] = None
        data = json.dumps(data)
        resp = Response(data, status=500, mimetype="application/json")
        end = time.time()
        print(f"Time taken {end-start:.3f} secs")
        return resp


if __name__ == "__main__":

    app.run(host="0.0.0.0", port=8081, use_reloader=False, debug=True, threaded=True)
