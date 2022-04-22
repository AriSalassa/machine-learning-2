import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
import category_encoders as ce
from geopy.geocoders import Nominatim


"""
def one_hot_encoding(X_in: pd.core.frame.DataFrame):
    assert type(X_in) == pd.core.frame.DataFrame
    columns = X_in.columns
    for column in columns:
        column_encoded = pd.get_dummies(X_in[column], prefix=column)
        X_in = X_in.drop(column, axis=1)
        X_in = X_in.join(column_encoded)

    return X_in
"""

class Transformer():
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.ohe = None
        self.be = None
        self.pt = None



    def extract_month_from_date(self, X_in: pd.core.frame.DataFrame, drop: bool = True, col_in_name: str = 'Date', col_out_name: str = 'Month'):
        X_in[col_out_name] = pd.DatetimeIndex(X_in[col_in_name]).month
        if drop:
            X_in.drop(col_out_name, axis=1, inplace=True)
        
        return



    def simple_class_encoding(self, X_in: pd.core.frame.DataFrame, label: str, dict: dict):
        assert type(X_in) == pd.core.frame.DataFrame
        mapped = X_in[label].map(dict)

        X_in[label] = mapped

        return X_in



    def decompose_angle_in_components(self, X_in: pd.core.frame.DataFrame, col_name: str):
        assert type(X_in) == pd.core.frame.DataFrame

        return np.cos(X_in[col_name]), np.sin(X_in[col_name])

    

    def get_geolocations(self, X_in: pd.core.frame.DataFrame, col_name: str = 'Location'):
        assert type(X_in) == pd.core.frame.DataFrame
        
        geolocator = Nominatim(timeout=10, user_agent = "test_app")
        cities = X_in[col_name].unique()

        city_location_dict = {}
        cities_no_found = []

        for city in cities:
            location = geolocator.geocode(re.sub(r"(?<![A-Z])(?<!^)([A-Z])",r" \1", city) + ' AU')
            if location is None:
                city_location_dict[city] = None
                cities_no_found.append(city)
            else:
                city_location_dict[city] = (location.latitude, location.longitude)
        
        return city_location_dict, cities_no_found



    def decompose_location_in_geolocations(self, X_in: pd.core.frame.DataFrame, dict_map: dict, col_name: str = 'Location'):
        assert type(X_in) == pd.core.frame.DataFrame
        
        lat_serie = X_in.apply(lambda row : dict_map[row[col_name]][0], axis = 1)
        lon_serie = X_in.apply(lambda row : dict_map[row[col_name]][1], axis = 1)

        return lat_serie, lon_serie



    def one_hot_encoding(self, X_in: pd.core.frame.DataFrame):
        assert type(X_in) == pd.core.frame.DataFrame

        if self.ohe is None:
            self.ohe = OneHotEncoder(
                categories="auto",
                drop=None,  # first/last=k-1,False = k (OHE)
                sparse=False,  # No devolver una matriz esparsa
                handle_unknown="ignore",
            )

            self.ohe.fit(X_in.fillna("Missing").values)
        
        cat_names = np.concatenate(self.ohe.categories_).ravel()

        X_in_tranformed = self.ohe.transform(X_in.fillna("Missing").values)
        X_out = pd.DataFrame(X_in_tranformed, columns=cat_names).astype(int)

        return X_out    



    def binary_encoding(self, X_in: pd.core.frame.DataFrame):
        assert type(X_in) == pd.core.frame.DataFrame
        
        columns = X_in.columns
        if self.be is None:
            self.be = ce.BinaryEncoder(cols = columns, handle_unknown='unknown', handle_missing='missing')
            self.be = self.be.fit(X_in)

        columns_encoded = self.be.transform(X_in)
        columns = X_in.columns
        encoder = ce.BinaryEncoder()
        for column in columns:
            column_encoded = encoder.fit_transform(X_in[column])
            X_in = X_in.drop(column, axis=1)
            X_in = X_in.join(column_encoded)

        return X_in


    def power_transform(self, X_in: pd.core.frame.DataFrame, method: str = 'yeo-johnson'):
        assert type(X_in) == pd.core.frame.DataFrame
        
        columns = X_in.columns
        if self.pt is None:
            self.pt = PowerTransformer(method=method)
            self.pt = self.pt.fit(X_in)

        data_out = self.pt.transform(X_in)

        X_out = pd.DataFrame(data=data_out, columns=columns)

        return X_out


