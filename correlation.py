import pandas as pd
from sklearn.ensemble import RandomForestClassifier
#from pyspark.ml.classification import RandomForestClassifier as rfc_sp
#from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F
import pyspark

class CleanCorrelation:

    def __init__(self, corr_matrix:pd.DataFrame, threshold:float=0.95):
        self.corr_matrix = corr_matrix
        self.threshold = threshold
        self.pairs = self.get_pairs(self.threshold)
        self.correlated_groups = self.get_correlated_groups(self.threshold)
        self.feats_correlated = sorted(list(self.get_unique_corr_feats(self.threshold)))
        self.feats_not_correlated = sorted(list(set(self.corr_matrix.columns).difference(self.feats_correlated)))


    def get_unique_corr_feats(self,threshold):
        '''Return all the features involved in a high-correlation pair'''
        pairs = self.get_pairs(threshold)
        unique_feats = set(list(pairs.feat1.unique()) + list(pairs.feat2.unique()))
        return unique_feats


    def get_pairs(self, threshold):
        '''Get the feature pairs that overpass a correlation threshold'''
        pairs =  self.corr_matrix.abs().unstack().sort_values(ascending=False)
        pairs = pairs.reset_index()
        pairs.columns = ['feat1','feat2','correlation']
        high_pairs =  pairs[(pairs.correlation >= threshold) & (pairs.feat1 != pairs.feat2)]
        return high_pairs
    

    def get_correlated_groups(self, threshold) -> dict:
        '''Group each feature with its most correlated variables'''
        grouped_features = []
        correlated_groups = []
        pairs = self.get_pairs(threshold)
        for feature in pairs.feat1.unique():
            if feature not in grouped_features:
                correlated_block = pairs[pairs.feat1 == feature]
                grouped_features = grouped_features + list(correlated_block.feat2.unique()) + [feature]
                correlated_groups.append(correlated_block)
        print(f'Found {len(correlated_groups)} correlated feature groups')
        print(f'out of {self.corr_matrix.shape[0]} total features')
        return correlated_groups


    def _select_rfc(self, group_df, abt: pd.DataFrame, target_col):
        '''Keep the most predictive feature of a group based on a Random Forest'''
        eval_feats = group_df.feat2.to_list() + [group_df.feat1.unique()[0]]
        rf = RandomForestClassifier(n_estimators=20, random_state=101, max_depth=4)
        rf.fit(abt[eval_feats].fillna(0), abt[target_col])
        # Feature importance using RFC
        importance = pd.Series(rf.feature_importances_, index=eval_feats)
        importance = importance.sort_values(ascending=False)
        print('---------')
        print(importance)
        return importance.index[0]


    def _select_less_zeros(self, group_df, abt: pyspark.sql.dataframe.DataFrame):
        '''Keep the variable with less zero or null values'''
        eval_feats = group_df.feat2.to_list() + [group_df.feat1.unique()[0]]
        zeros_count_cols = [F.count(F.when((F.col(i) == 0) | (F.col(i).isNull()),1)).alias(i) for i in eval_feats]
        zeros = abt.select(*zeros_count_cols)
        pd_zeros = zeros.toPandas().T.reset_index(names = 'feature').rename(columns={0:'zeros'}).sort_values('zeros')
        print('---------')
        print(pd_zeros)
        return pd_zeros.iloc[0,0]


    def clean_with_rf(self, abt:pd.DataFrame, target_col):
        '''Clean correlated groups using RandomForest'''
        selected = []
        for group in self.correlated_groups:
            selected.append(self._select_rfc(group, abt, target_col))
        return selected


    def clean_with_zeros(self, abt: pyspark.sql.DataFrame):
        '''Clean correlated groups counting zeros and nulls for each column'''
        selected = []
        for group in self.correlated_groups:
            selected.append(self._select_less_zeros(group, abt))
        return selected



    def __pending(self,group_df, abt:pyspark.sql.DataFrame, target_col):
        '''PENDIENTE
        Keep the most predictive feature of a group based on a Random Forest
        eval_feats = group_df.feat2.to_list() + [group_df.feat1.unique()[0]]
        print(eval_feats)
        assembler = VectorAssembler(inputCols=eval_feats, outputCol="features")
        assembled_abt = assembler.transform(abt.select(target_col,*eval_feats))

        #Train
        mod_rfc = rfc_sp(numTrees=20, maxDepth=4, labelCol=target_col, seed=101)
        model = mod_rfc.fit(assembled_abt)
        # Feature importance
        importance = model.featureImportances.values
        print(importance)
        imp_df = pd.DataFrame({'feature':eval_feats,
                            'importance':importance}).sort_values('importance',ascending=False)
        print('---------')
        print(imp_df)
        print('---------')
        selected_feature = imp_df.iloc[0,0]
        return selected_feature'''
        pass #En algunos casos predice la misma proba para todos los registros