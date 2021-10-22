import http
from flask_cors import CORS, cross_origin
from flask import request, Flask, Response, jsonify, abort
import geopy
import json

app = Flask(__name__)

@app.route('/normalise', methods=['POST'])
def main():

    return json.dumps(http.HTTPStatus.OK).encode('utf8')


if __name__ == '__main__':
    app.run('0.0.0.0', port=2000)
