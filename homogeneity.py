df_pivote = spark.read.parquet(path_score_grupos).cache()
print("Total de registros para las estrategias {:,} con {} columnas".format(df_pivote.count(),len(df_pivote.columns)))
import numpy as np
from scipy import stats
def test_homogeidad(test = "ks"):
    tests =  { "ks" : stats.ks_2samp \
              ,"kruskal" : stats.kruskal \
              ,"mannwhitneyu" : stats.mannwhitneyu }
    
    # ks :iguldad de distros
    # kruskal : igualdad de medianas
    # mann whitney u :  igualdad de distros
    
    # pval > 0.05 -> iguales
    return tests[test]
def homogeneidad(sample1 , sample2 , tests):
    df=pd.DataFrame()
    for test in tests:
        value_s , p_value = test_homogeidad(test)(sample1 , sample2)
        df= df.append( pd.DataFrame({"test" : [test] ,  "statistic": [value_s] ,  "p_value": [p_value]}))
    return df
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