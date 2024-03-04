import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from pyspark.sql import functions as F
import pyspark
import seaborn as sns
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
import matplotlib.pyplot as plt

class CleanCorrelation:

    def __init__(self, corr_matrix:pd.DataFrame, threshold:float=0.95):
        self.corr_matrix = corr_matrix
        self.threshold = threshold
        self.pairs = self.get_pairs(self.threshold)
        self.correlated_groups = self.get_correlated_groups(self.threshold)
        self.feats_correlated = sorted(list(self.get_unique_corr_feats(self.threshold)))
        self.feats_not_correlated = sorted(list(set(self.corr_matrix.columns).difference(self.feats_correlated)))

    
    @classmethod    
    def heatmap(cls, corr_matrix, height = 10, width = 10):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(height, width))
        sns.heatmap(corr_matrix, annot = True, cbar = False, 
                    annot_kws = {"size": 8}, vmin = -1, vmax = 1, center = 0,
                    cmap = sns.diverging_palette(20, 220, n=200),square = True,ax = ax)
        ax.set_xticklabels(ax.get_xticklabels(),rotation = 45,horizontalalignment = 'right')
        ax.tick_params(labelsize = 10)


    @classmethod
    def get_correlation(cls, df:pyspark.sql.dataframe.DataFrame, drop=[]):
        df2 = df.drop(*drop) if drop else df
        features = df2.columns
        assembler = VectorAssembler(inputCols=features, outputCol='features')
        df_vector = assembler.transform(df2).select('features')
        matrix = Correlation.corr(df_vector, 'features').collect()[0][0] 
        corr_matrix = matrix.toArray().tolist() 
        corr_matrix_df = pd.DataFrame(data=corr_matrix, columns = features, index=features) 
        return corr_matrix_df


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

    
    def clean_with_one_side(self):
        '''Keep all the columns from one side of the highly correlated pairs'''
        all_feats = self.corr_matrix.columns
        pairs = self.corr_matrix.unstack().sort_index().reset_index().rename(columns={0:'corr'})
        
        #Quitar pares duplicados y diagonal orig
        out = pairs[~pairs[['level_0', 'level_1']].apply(frozenset, axis=1).duplicated()]
        out = out[out.level_1 != out.level_0]
        out['abs_corr'] = out['corr'].apply(lambda x: abs(x))
        out = out[out.abs_corr >= self.threshold]
        
        #Variables unicas con altas correlaciones
        highly_correlated = list(set(out.level_0.to_list() + out.level_1.to_list()))
        print(f'Hay {out.shape[0]} pares que superan el {self.threshold} de correlacion y contemplan  {len(highly_correlated)} distintas variables')
        feats_no_corr = [i for i in all_feats if i not in highly_correlated]
        
        #Conservar 1 variable de cada par con alta correlacion
        preserve_corr = list(set(out.level_0.to_list()))
        final_preserve = [i for i in preserve_corr if i not in set(out.level_1.to_list())]
        
        final_feats = feats_no_corr + final_preserve
        print(f'Se agregaron {len(final_preserve)} variables a las {len(feats_no_corr)} que no superaban el {self.threshold} de correlacion\n')
        print(f'VARIABLES FINALES:  {len(final_feats)}')
        return final_feats,out
    
        
    def clean_with_rank(self, predictors2,corr_matrix_df): #Funcion de JC
        'Excluye variables que esten correlacionadas con una de mayor importancia'
        try:
            exclude=[]
            for j,col_i in enumerate(predictors2[0:20]):
                if col_i in exclude+["segmento_risk","SEG_VIDA"]:
                    print("Ignoring ",col_i)
                    continue
                print(corr_matrix_df.shape)
                corr_matrix_df_i=corr_matrix_df[ (corr_matrix_df.index.isin(predictors2[0:j+1]) )==False].copy()
                print(corr_matrix_df_i.shape)
                corr_matrix_df_i=corr_matrix_df[[col_i]].query(f"abs({col_i}) >= 0.95").copy()
                print(corr_matrix_df_i.shape)
                corr_matrix_df_i.reset_index(inplace=True)
                ex_i=list(corr_matrix_df_i["index"])
                ex_i.remove(col_i)
                exclude+=ex_i
                return exclude
        except:
            pass


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