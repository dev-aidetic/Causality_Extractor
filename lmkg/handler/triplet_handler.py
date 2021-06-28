from ..relationship_engine import relationship_finder
from flask import jsonify
from ..utils import utils

logger = utils.intialize_logging(__name__)

relationship_finder = relationship_finder.RelationshipFinder()


def relationship_handler(text):
    """
    if "text" in request_form:
        text = request_form["text"]
    else:
        raise Exception("Add text field in request")
    """

    result = relationship_finder.find_from_text(text)
    print(result)

    return result
