import pandas as pd
import threading
import numpy as np
from scipy import stats
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

class Homogeneity:
    
    def __init__(self,base:dict,tests:dict,compare_cols:list[str],methods_limits:dict = {'ks': 0.05,
                                                                                         'kruskal': 0.05,
                                                                                         'mannwhitneyu': 0.05,
                                                                                         'levene': 0.05,
                                                                                         'mean': 0.05,
                                                                                         'quartiles':0.05}) -> None:
        '''
        base: Dictionary with the alias and dataframe for the base sample
        tests: Dictionary with the aliases and dataframes for each test sample
        compare_cols: Columns to be compared between base and each test sample
        methods: List of methods to test the samples on. 
        ks -> p_value for same cumulative distribution
        kruskal -> p_value for same median
        mannwhitneyu -> p_value for same distribution
        mean -> relative difference vs base mean
        quartiles -> max relative difference vs base quartiles 
        '''
        self.base = base
        self.tests = tests
        self.compare_cols = compare_cols
        self.methods = list(methods_limits.keys())
        self.methods_limits = methods_limits

        #Evaluation and summary
        self.tests_detail = self.run_samples_tests(self.base, self.tests, self.compare_cols)
        self.summary_features = self.get_summary_by_feature(self.tests_detail,self.compare_cols)
        self.summary_all = self.get_overall_summary(self.tests_detail)

    

    def _compare_mean(self,sample1:pd.Series, sample2:pd.Series):
        '''Get the relative difference of two means'''
        mean1 = np.mean(sample1)
        mean1 = mean1 if mean1 != 0 else 0.001
        mean2 = np.mean(sample2)
        return None, round(abs(mean2/mean1-1),2)

    def _get_quartiles(self,sample1:pd.Series,sample2:pd.Series):
        qtls = [0.25, 0.50, 0.75]
        qt_vals1 = sample1.quantile(qtls).to_list()
        qt_vals2 = sample2.quantile(qtls).to_list()
        return qt_vals1, qt_vals2
    

    def _compare_quartiles(self,sample1:pd.Series,sample2:pd.Series):
        '''Get the maximum relative difference between quartiles in two data samples'''
        #Quartiles for each sample
        qt_vals1, qt_vals2 = self._get_quartiles(sample1,sample2)
        qt_vals1 = [i if i!=0 else 0.001 for i in qt_vals1]         
        #Relative differences
        diffs = [abs(qs2/qs1-1) for qs1,qs2 in zip(qt_vals1,qt_vals2)]
        return None, round(max(diffs) ,2)


    def _test_methods(self,test):  
        '''Available methods to test the homogeneity on the data'''
        tests =  {"ks":stats.ks_2samp, #Same cumulative distribution
                  "kruskal":stats.kruskal, #Same median
                  "mannwhitneyu":stats.mannwhitneyu, #Same distribution
                  'levene':stats.levene,
                  'mean': self._compare_mean,
                  'quartiles': self._compare_quartiles}

        # pval greater than 0.05 then homogeneous
        return tests[test]


    def compare_samples(self, base: pd.Series, test: pd.Series, methods_limits=None):
        '''Evaluate the homogeneity in 2 data samples'''
        dfs = []
        methods_limits = methods_limits if methods_limits is not None else self.methods_limits

        def ix_passed(test_name, decision_value):
            '''Decide whether the samples passed the test'''
            ix_mapping = {'ks': decision_value > methods_limits[test_name],
                          'kruskal': decision_value > methods_limits[test_name],
                          'mannwhitneyu': decision_value > methods_limits[test_name],
                          'levene': decision_value > methods_limits[test_name],
                          'mean': decision_value <= methods_limits[test_name],
                          'quartiles': decision_value <= methods_limits[test_name]}
            return int(ix_mapping[test_name])

        def process_method(method):
            'Single evaluation of a statistical test'
            value_s, decision_value = self._test_methods(method)(base, test)
            result_method = pd.DataFrame({"method": method,
                                           "statistic": value_s,
                                           'mean_sample_base': base.mean(),
                                           'mean_sample_test': test.mean(),
                                           "decision_value": decision_value}, index=[0])
            result_method['passed'] = result_method.apply(lambda x: ix_passed(x['method'], x['decision_value']),axis=1)
            with threading.Lock():
                dfs.append(result_method)

        threads = [threading.Thread(target=lambda m=method: process_method(m)) for method in methods_limits.keys()]
        [t.start() for t in threads]    
        [t.join() for t in threads]    

        return pd.concat(dfs)


    def run_samples_tests(self, base, tests:dict, cols:list[str], methods_limits = None) -> pd.DataFrame:
        '''Evaluate homogeneity over multiple dataframes and features
        base: DataFrame
        tests: dictionary with the aliases and dataframes for each test sample
        cols: list of columns to consider for the tests
        methods_limits: Acceptance thresholds for each test
        '''
        dfs = []
        methods_limits = methods_limits if methods_limits != None else self.methods_limits
        #Function to parallelize
        def process_feature(test_alias, test_df, col):
            sample_len = test_df.shape[0]
            base_col = base[col]
            test_col = test_df[col]
            feat_results = self.compare_samples(base_col, test_col, methods_limits)
            feat_results['sample_alias'] = test_alias
            feat_results['column'] = col
            feat_results['sample_size'] = sample_len
            #Guarantee appropriate append
            with threading.Lock():
                dfs.append(feat_results)

        #Each dataframe
        for test_alias, test_df in tests.items():
            print(f'Working on {test_alias} sample')
            #Create the threads
            threads = [threading.Thread(target=process_feature, args=(test_alias, test_df, col)) for col in cols]
            [t.start() for t in threads]
            [t.join() for t in threads]
        
        sort_cols = ['sample_alias','column','method','sample_size',
                    'mean_sample_base','mean_sample_test',
                    'statistic','decision_value','passed']
        df_fin = pd.concat(dfs).reset_index(drop=True)[sort_cols]
        return df_fin    
    

    def get_overall_summary(self, tests_detail):
        '''Get the overall summary of the tests'''
        return tests_detail.pivot_table(index='column',
                                        columns='sample_alias',
                                        values='passed',
                                        aggfunc='mean',
                                        margins=True,
                                        margins_name='pass_rate').round(3)
    

    def get_summary_by_feature(self, tests_detail:pd.DataFrame, relevant_features:list[str]):
        '''Get a summary of a selection of features'''
        return tests_detail[tests_detail['column'].isin(relevant_features)].pivot_table(index='column',
                                                                                    columns=['sample_alias','method'],
                                                                                    values='passed',
                                                                                    aggfunc='mean',
                                                                                    margins=True,
                                                                                    margins_name='pass_rate').round(3)

    
    ###
    ### SEQUENTIAL FUNCTIONS (ONLY FOR TESTING PURPOSES)
    ###

    def __compare_samples_seq(self,base:pd.Series, test:pd.Series , methods_limits = None):
        '''Evaluate the homogeneity in 2 data samples'''
        methods_limits = methods_limits if methods_limits != None else self.methods_limits
        def ix_passed(test_name, decision_value):
            '''Decide whether the samples passed the test'''
            ix_mapping =  {'ks': decision_value > methods_limits[test_name],
                           'kruskal': decision_value > methods_limits[test_name],
                           'mannwhitneyu':decision_value > methods_limits[test_name],
                           'levene': decision_value > methods_limits[test_name],
                           'mean': decision_value <= methods_limits[test_name],
                           'quartiles': decision_value <= methods_limits[test_name]}
            return int(ix_mapping[test_name])

        dfs=[]
        for method in methods_limits.keys(): #SEQUENTIAL PART
            value_s , decision_value = self._test_methods(method)(base , test)
            result_method = pd.DataFrame({"method" : method,
                                           "statistic": value_s,
                                           'mean_sample_base': base.mean(),
                                           'mean_sample_test': test.mean(),
                                           "decision_value": decision_value}, index = [0])
            result_method['passed'] = result_method.apply(lambda x: ix_passed(x['method'],x['decision_value']),axis=1)
            dfs.append(result_method)
        return pd.concat(dfs)

    
    def __run_samples_tests_seq(self, base, tests:dict, cols:list[str], methods_limits = None) -> pd.DataFrame:
        '''same as above but without parallelizing. Only for test purposes'''
        dfs = []
        methods_limits = methods_limits if methods_limits != None else self.methods_limits

        #Each dataframe
        for test_alias, test_df in tests.items():
            print(f'Working on {test_alias} sample')
            for col in cols: #SEQUENTIAL PART
                sample_len = test_df.shape[0]
                base_col = base[col]
                test_col = test_df[col]
                feat_results = self.compare_samples_seq(base_col, test_col, methods_limits)
                feat_results['sample_alias'] = test_alias
                feat_results['column'] = col
                feat_results['sample_size'] = sample_len
                dfs.append(feat_results)
        
        sort_cols = ['sample_alias','column','method','sample_size',
                    'mean_sample_base','mean_sample_test',
                    'statistic','decision_value','passed']
        df_fin = pd.concat(dfs).reset_index(drop=True)[sort_cols]
        return df_fin    
    



