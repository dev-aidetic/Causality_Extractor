# settings.py
import json
import os
from dotenv import load_dotenv
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(verbose=True)


class EnvVariableNotSet(Exception):
    pass


def check_env_vars(env):
    for v in dir(env):
        if getattr(env, v) is None and not v.startswith("__"):
            print("ENV VARIABLES LOAD: FAIL")
            raise EnvVariableNotSet(f"{v} NOT SET!")
    print("ENV VARIABLES LOAD: SUCCESS")


class _Env:
    def __init__(self):
        pass


env = _Env()

check_env_vars(env)
