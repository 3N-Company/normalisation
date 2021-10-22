from outlier import SmallUser, Normalisation
import http
from flask_cors import CORS, cross_origin
from flask import request, Flask, Response, jsonify, abort
import geopy
import pandas as pd
import json


app = Flask(__name__)




def get_position_info(self, data: dict):
    latitude = data['metadata']["position"]['latitude']
    longitude = data['metadata']["position"]['longitude']
    return latitude, longitude

def provide_score_for_user(self):
    """
    # todo provide a score for a user
    :return:
    """
    pass


def iqr_outliers():
    pass






def remove_outliers(data):
    annotation_list = []
    normalis = Normalisation()


    for elem in data:
        user_data = SmallUser(elem)
        normalis.user_array.append(user_data.user)
        normalis.latitude_array.append(user_data.latitude)
        normalis.longitude_array.append(user_data.longitude)
        normalis.score_array.append(user_data.score)

    d = {'user': normalis.user_array, 'latitude': normalis.latitude_array, "longitude":normalis.longitude_array,
         "score":normalis.score_array}
    df = pd.DataFrame(data=d)



    df = normalis.process_zscore(df, "longitude")
    df = normalis.process_zscore(df, "latitude")

    # IQR FILTER
    upper_bound_lat, lower_bound_lat = normalis.iqr_lower_uperbound_calculation(df, "latitude")
    upper_bound_long, lower_bound_long = normalis.iqr_lower_uperbound_calculation(df, "longitude")
    df['latitude'] = df['latitude'].map(lambda x: normalis.lqr_help_func(x, upper_bound_lat, lower_bound_lat))
    df['longitude'] = df['longitude'].map(lambda x: normalis.lqr_help_func(x, upper_bound_long, lower_bound_long))

    df['score'] = ~(df.isnull().any(axis=1))
    recommend_json = df.to_json(orient="index", force_ascii=False)
    recommend_json_with_slashes = json.dumps(json.loads(recommend_json), ensure_ascii=False).encode('utf8')
    return recommend_json_with_slashes


@app.route('/normalise', methods=['POST'])
def main():
    data = request.get_json()
    if len(data) > 1:
        ret = remove_outliers(data)

    return ret


if __name__ == '__main__':
    app.run('0.0.0.0', port=2000)
