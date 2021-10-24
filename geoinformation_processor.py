from outlier import SmallUser, Normalisation
import http
from flask_cors import CORS, cross_origin
from flask import request, Flask, Response, jsonify, abort
import geopy
import pandas as pd
import json
from typing import List, Dict

app = Flask(__name__)

class Filter:

    def __init__(self):
        self.normalis = Normalisation()

    def prepare_df_to_filter(self, data:List[dict]) -> pd.DataFrame:
        """
        Preparing a pandas Dataframe for further
        Args:
            data: sended over rest list of jsons

        Returns:
            filled pandas dataframe
        """
        for elem in data:
            user_data = SmallUser(elem)
            self.normalis.user_array.append(user_data.id)
            self.normalis.latitude_array.append(user_data.latitude)
            self.normalis.longitude_array.append(user_data.longitude)
            self.normalis.score_array.append(user_data.score)

        d = {'id': self.normalis.user_array, 'latitude': self.normalis.latitude_array, "longitude": self.normalis.longitude_array,
             "score": self.normalis.score_array}
        df = pd.DataFrame(data=d)
        return df


    def remove_outliers(self, data: List[dict]) -> bytes:
        """
        Removing outliers by using
            1. z-score
            2. iqr filter
        Args:
            data: sended over rest list of jsons

        Returns:
            returns a json in a form of a scoring list with ids and boolean value, if the proposed information
            by user was helpful(was not thrown away). and the mean value of the coordinates for the latitude
            longitude

            Examples:
                 {"mean": {"latitude": "54.409669", "longitude": "23.976294"}, "user": [{"id": "fang", "score": true}]}
        """
        df = self.prepare_df_to_filter(data)
        df = self.normalis.process_zscore(df, "longitude")
        df = self.normalis.process_zscore(df, "latitude")

        # IQR FILTER
        upper_bound_lat, lower_bound_lat = self.normalis.iqr_lower_uperbound_calculation(df, "latitude")
        upper_bound_long, lower_bound_long = self.normalis.iqr_lower_uperbound_calculation(df, "longitude")
        df['latitude'] = df['latitude'].map(lambda x: self.normalis.lqr_help_func(x, upper_bound_lat, lower_bound_lat))
        df['longitude'] = df['longitude'].map(lambda x: self.normalis.lqr_help_func(x, upper_bound_long, lower_bound_long))


        df['score'] = ~(df.isnull().any(axis=1))
        frame_to_send = df[['id', 'score']]
        mean_values = df[(df['latitude'].notna()) & (df['longitude'].notna())][['latitude', 'longitude']]

        reply, mean = {}, {}
        reply["mean"] = mean
        mean['latitude'] = str(mean_values[['latitude']].mean().tolist()[0])
        mean['longitude'] = str(mean_values[['longitude']].mean().tolist()[0])
        score_json = frame_to_send.to_json(orient='records', force_ascii=False)
        reply['user'] = json.loads(score_json)

        filtered_data_with_score = json.dumps(reply, ensure_ascii=False).encode('utf8')
        return filtered_data_with_score


@app.route('/normalise', methods=['POST'])
def main():
    data = request.get_json()
    filter = Filter()
    ret = filter.remove_outliers(data)

    return ret


if __name__ == '__main__':
    app.run('0.0.0.0', port=2000)
