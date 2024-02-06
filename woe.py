import config_pc
import math
import json
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import col


class WOE_IV:

    def __init__(self, df: DataFrame, cols_to_woe: [str], label_column: str, good_label: int):
        self.df = df
        self.cols_to_woe = cols_to_woe
        self.label_column = label_column
        self.good_label = good_label
        self.fit_data = {}

    def fit(self):
        agg_class = self.df.groupby((col(self.label_column) == self.good_label).alias('ix_good_label')).count().cache()
        total_good = agg_class.filter(F.col('ix_good_label') == 'true').select('count').collect()[0][0]
        total_bad = agg_class.filter(F.col('ix_good_label') == 'false').select('count').collect()[0][0]
        agg_class.unpersist()

        for col_to_woe in self.cols_to_woe:
            print(f'Trabajando en categoria {col_to_woe}')
            ag_col = self.df.withColumn('ix_good_label',(col(self.label_column) == self.good_label))\
                            .groupby(col_to_woe).pivot('ix_good_label', values=['true','false']).count().fillna(0).cache()
            
            categories = ag_col.select(col_to_woe).collect()
            categories = [i[0] for i in categories]

            for cat_value in categories:
                row_cat_label = ag_col.filter((col(col_to_woe) == cat_value)).collect()[0]
                
                good_amount = row_cat_label['true']
                bad_amount = row_cat_label['false']

                good_amount = good_amount if good_amount != 0 else 0.5
                bad_amount = bad_amount if bad_amount != 0 else 0.5
                
                good_dist = good_amount / total_good
                bad_dist = bad_amount / total_bad

                self.build_fit_data(col_to_woe, cat_value, good_dist, bad_dist)

            ag_col.unpersist()

    def build_fit_data(self, col_to_woe, category, good_dist, bad_dist):
        woe_final = math.log(good_dist / bad_dist)
        woe_info = {category: {'woe': woe_final,
                                'iv': (good_dist - bad_dist) * woe_final}}
        if col_to_woe not in self.fit_data:
            self.fit_data[col_to_woe] = woe_info
        else:
            self.fit_data[col_to_woe].update(woe_info)

    def transform(self, df: DataFrame, drop_original = False):
        for col_to_woe, woe_info in self.fit_data.items():
            itera_woe_info =  F.coalesce(*[F.when(F.col(col_to_woe) == category, F.lit(woe_iv['woe'])) for category, woe_iv in woe_info.items()],F.lit(0))
            df = df.withColumn(col_to_woe + '_woe', itera_woe_info)
        if drop_original:
            return df.drop(*self.fit_data.keys())
        return df
    
    def save_params(self, path):
        with open(path,'w') as file:
            json.dump(self.fit_data, file)
        print('Datos guardados')

    def load_params(self, path):
        with open(path,'r') as file:
            self.fit_data = json.load(file)
        print('Datos importados')