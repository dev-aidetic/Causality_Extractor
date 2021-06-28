from flask import Flask, request
import logging
from traceback import print_exc

from settings import env
from handler import triplet_handler
from utils.utils import intialize_logging


logger = intialize_logging(__name__)

app = Flask("LMKG")


@app.route("/lmkg_triplets", methods=["POST"])
def lmkg_triplets():

    try:
        request_form = request.get_json(force=True)
        logger.info(f"Request Format: {request_form}")

        triplet_result = triplet_handler.relationship_handler(request_form)

        return triplet_result
    except Exception as e:
        logger.error(e)
        logger.debug(print_exc())
        return "Error while processing the request. Check the request."


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8002, debug=False)
