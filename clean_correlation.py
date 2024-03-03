import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class CleanCorrelation:

    def __init__(self, corr_matrix:pd.DataFrame, threshold:float=0.95):
        self.corr_matrix = corr_matrix
        self.threshold = threshold
        self.pairs = self.get_pairs(self.threshold)
        self.correlated_groups = self.get_correlated_groups(self.threshold)

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
        correlated_groups = {}
        pairs = self.get_pairs(threshold)
        for feature in pairs.feat1.unique():
            if feature not in grouped_features:
                correlated_block = pairs[pairs.feat1 == feature]
                grouped_features = grouped_features + list(correlated_block.feat2.unique()) + [feature]
                correlated_groups[feature] = correlated_block
        print(f'Found {len(correlated_groups)} correlated feature groups')
        print(f'out of {self.corr_matrix.shape[0]} total features')
        return correlated_groups


    def select_rfc(self, group_df, abt, target_col):
        '''Keep the most predictive feature of a group based on a Random Forest'''
        eval_feats = group_df.feat2.to_list() + [group_df.feat1.unique()[0]]
        rfc = RandomForestClassifier(n_estimators=20, random_state=101, max_depth=4)
        rfc.fit(abt[eval_feats].fillna(0), abt[target_col])
        # Feature importance using RFC
        importance = pd.Series(rfc.feature_importances_, index=eval_feats)
        importance = importance.sort_values(ascending=False)
        print('---------')
        print(importance)
        print('---------')
        return importance.index[0]

    def iterate_rfc(self, abt, target_col):
        '''Return the selected features over different high-correlation groups'''
        selected = []
        for main_feat, group_df in self.correlated_groups.items():
            selected.append(self.select_rfc(group_df, abt, target_col))
        return selected
