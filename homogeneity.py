import pandas as pd
import numpy as np
from scipy import stats


class Homogeneity:
    

    def_method_limits = {'ks':0.05,
                         'kruskal':0.05,
                         'mannwhitneyu':0.05,
                         'mean':0.05,
                         'quartiles':0.05}

    def __init__(self,base:dict,tests:dict, compare_cols:list[str], methods_limits = def_method_limits) -> None:
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
        #PONER RUN TESTS AQUI
        self.run_samples_tests(self.base, self.tests, self.compare_cols, )
    

    def _compare_mean(sample1:pd.Series, sample2:pd.Series):
        mean1 = np.mean(sample1)
        mean2 = np.mean(sample2)
        return None, abs(mean2/mean1-1)


    def _compare_quartiles(sample1:pd.Series,sample2:pd.Series):
            qtls = [0.25, 0.50, 0.75]
            #Quartiles for each sample
            qt_vals1 = sample1.quantile(qtls).to_list()
            qt_vals2 = sample2.quantile(qtls).to_list()
            #Relative differences
            diffs = [abs(qs2/qs1-1) for qs1,qs2 in zip(qt_vals1,qt_vals2)]
            return None, max(diffs) 
    

    def _test_methods(self,test):  
        '''Available methods to test the homogeneity on the data'''
        tests =  {"ks":stats.ks_2samp, #Same cumulative distribution
                  "kruskal":stats.kruskal, #Same median
                  "mannwhitneyu":stats.mannwhitneyu, #Same distribution
                  'mean': self._compare_mean,
                  'quartiles': self._compare_quartiles}

        # pval greater than 0.05 then homogeneous
        return tests[test]

 
    def compare_samples(self,base:pd.Series, test:pd.Series , methods_limits = def_method_limits):
        '''Evaluate the homogeneity in 2 data samples
        
        tests: ks -> same cumulative distribution
               kruskal -> same median
               mannwhitneyu -> same distribution
               mean -> same mean
               quartiles -> same quartiles '''

        def ix_passed(test_name, decision_value):
            '''Decide whether the samples passed the test'''
            ix_mapping =  {'ks': decision_value > methods_limits[test_name],
                           'kruskal': decision_value > methods_limits[test_name],
                           'mannwithneyu':decision_value > methods_limits[test_name],
                           'mean': decision_value <= methods_limits[test_name],
                           'quartiles': decision_value <= methods_limits[test_name]}
            return int(ix_mapping[test_name])

        df=pd.DataFrame()
        for method in methods_limits.keys():
            value_s , decision_value = self._test_methods(method)(base , test)
            result_method = pd.DataFrame({"method" : [method],
                                           "statistic": [value_s],
                                           'mean_sample_base': [base.mean()],
                                           'mean_sample_test': [test.mean()],
                                           "decision_value": [decision_value]})
            result_method['passed'] = result_method.apply(lambda x: ix_passed(x['method'],x['decision_value']))
            df= df.append(result_method)
        return df
    

    def run_samples_tests(self, base, tests, cols, methods_limits = def_method_limits):
        '''Evaluate homogeneity over multiple dataframes and features'''        
        
        result_df = pd.DataFrame()
        #Each dataframe
        for test_alias, test_df in tests.items():
            print(f'Working on {test_alias} sample')
            sample_len = test_df.shape[0]
            #Each feature
            for col in cols:
                print(f' --{col}')
                base_col = base[col]
                test_col = test_df[col]
                feat_results = self.compare_samples(base_col, test_col, methods_limits)
                feat_results['sample_alias'] = test_alias
                feat_results['column'] = col
                feat_results['sample_size'] = sample_len
                result_df.append(feat_results)
        return result_df















    def feat_tests(df , vars_ , sample1_cond, sample_dict , tests = ["ks"]):
        df_hf=pd.DataFrame()
        for var_i in vars_: 
            print("-------  {}  ------".format(var_i))
            print(" sample 1 cond : ", sample1_cond)
            sample1=pd.to_numeric(df.filter(sample1_cond).select(var_i).toPandas()[var_i])
            ctrl_mean = np.mean(sample1)
            for sample_key ,sample_i in sample_dict.items():
                print("-------  {}  ------".format(sample_key))
                print(" sample 2 cond : ", sample_i)
                sample2=pd.to_numeric(df.filter(sample_i).select(var_i).toPandas()[var_i])
                print("len s1 : ",len(sample1))
                print("mean s1 : ",ctrl_mean)
                print("---")
                print("len s2 : ",len(sample2))
                mean_s2=np.mean(sample2)
                print("mean s2 : ",mean_s2)
                    
                    
                df_h=homogeneidad(sample1 , sample2 , tests)
                
                df_h["sample"]=sample_key
                df_h["n_sample"] = len(sample2)
                df_h["n_control"] = len(sample1)
                df_h["sample_mean"] = mean_s2
                df_h["ctrl_mean"] = ctrl_mean
                df_h["feature"] = var_i
                
                df_hf=df_hf.append(df_h)
        return df_hf

    def all_tests(conds, vars_, tests = ["ks"]):
        final_df = pd.DataFrame()
        vars_ = [ x for x in vars_ if x not in ['discount']]
        df_hf= pd.DataFrame()
        for cond_d in conds : 
            df_cond = feat_tests(df_pivote , vars_ = vars_ , sample1_cond = cond_d["ctrl"] , sample_dict = cond_d["tests"] , tests = tests)
            df_hf=df_hf.append(df_cond)
        # df_hf.to_csv(os.path.join(path_write,date_+".csv"),index=False)
        final_df=final_df.append(df_hf)
        return final_df

    #Grupos control
    ctrl_completo = ((F.col("Grupo")=="Control") & (F.col("Estrategia")=="Exploratorio"))
    ctrl_top = ((F.col("Grupo")=="Control") & (F.col("Estrategia")=="Exploratorio") & ((F.col('vip_top_20_monto_est') == 1)|(F.col('vip_top_2_deciles_cont') == 1)))
    ctrl_normal = ((F.col("Grupo")=="Control") & (F.col("Estrategia")=="Exploratorio") & ~((F.col('vip_top_20_monto_est') == 1)|(F.col('vip_top_2_deciles_cont') == 1)))
    #Grupos aleatorios
    Rnd_cond= ((F.col("Grupo")=="Test") & (F.col("Estrategia")=="Exploratorio"))
    rnd_dict = {"Rnd": Rnd_cond}
    rnd_dict_d= {"Rnd_{}".format(i): ((F.col("Grupo")=="Test") & (F.col("Estrategia")=="Exploratorio") & (F.col("Descuento")==i)) for i in DISCOUNTS if i!=0}
    rnd_dict.update(rnd_dict_d)
    #Optimizado
    opti_completo = {'Optimizado_completo' : (F.col("vip_grupo_final").isin(["Test",'BAU']) & (F.col("vip_estrategia_final").isin(["Etiqueta",'Optimizado'])))}
    opti_top = {'Etiqueta' : (F.col("vip_grupo_final")=="Test") & (F.col("vip_estrategia_final")=="Etiqueta")}
    opti_normal = {'Optimizado': (F.col("vip_grupo_final")=="BAU") & (F.col("vip_estrategia_final")=="Optimizado")}
    #Comparar control vs todos los tests (desc: 5,10,15,20)
    conds= [{"ctrl" : ctrl_completo, "tests": rnd_dict|opti_completo}]
    
    
    final_df = all_tests(conds = conds, vars_ = vars_pruebas, tests = ["ks","kruskal","mannwhitneyu"])
    final_df['status'] = np.where(final_df['p_value']<=0.05, 0, 1)
    homogein_detail = pd.pivot_table(final_df, index = ['feature'], columns = ['sample'], values = 'status', aggfunc= lambda x : 1 if x.median() == 1 else 0)
    homogein_results = homogein_detail.apply(lambda x : x.median(),axis=1)
    print(f'Homogein ratio : {homogein_results.sum() / homogein_results.shape[0]}')
    homogein_results.sort_values()