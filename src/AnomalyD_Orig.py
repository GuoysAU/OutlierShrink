from Package import *
import argparse


def TestWindowSize(pathList): 
    filenamesList = [Filenames2List(path) for path in pathList]
    for i, f in enumerate(filenamesList):
        path = pathList[i]
        print(f"Dataset: {path.split('data')[1].split('/')[1]}")
        for index, filename in enumerate(f):
            # 1. Data Processing
            df = pd.read_csv(path+filename , header=None).dropna().to_numpy()
            max_length = df.shape[0]
            data = df[:max_length,0].astype(float)
            label = df[:max_length,1].astype(int)
            if((list(label).count(1))==0):
                print(f"{index+1}.{filename[:18]:<18}: No Anomoly")
                continue
            slidingWindow = find_length(data)
            print(f"Window Size = {slidingWindow }, Data Size = {max_length}")
        print("")

def run(pathList, model, output_file, random_state=None): 
    def log(message):
        print(message, flush=True)  
        output_file.write(message + "\n") 
        output_file.flush()   
    log("==================================================")
    log(f"Starting analysis for Model: {model}")
    log("==================================================")

    filenamesList = [Filenames2List(path) for path in pathList]
    for i, f in enumerate(filenamesList):
        path = pathList[i]
        ROC_mean, PR_mean, F_mean, Time_mean, n = 0, 0, 0, 0, 0
        epistemics = []
        log(f"Dataset: {path.split('data')[1].split('/')[1]}")
        for index, filename in enumerate(f):
            # 1. Data Processing
            df = pd.read_csv(path+filename , header=None).dropna().to_numpy()
            max_length = df.shape[0]
            data = df[:max_length,0].astype(float)
            label = df[:max_length,1].astype(int)
            if((list(label).count(1))==0):
                log(f"{index+1}.{filename[:18]:<18}: No Anomoly")
                continue
            slidingWindow = find_length(data)
            #****************************************#
            if(model=="LOF"):
                slidingWindow  = 1
            #****************************************#
             #****************************************#
            if(model=="IForest"):
                slidingWindow = 1
            #****************************************#

            X_data = Window(window = slidingWindow).convert(data).to_numpy()


            clf_ori = IForest(n_jobs=1, contamination=0.1, random_state=random_state)
            clf_ori.fit(X_data)
            epistemic = clf_ori.epistemic_uncertainty
            epistemic_all = epistemic
            epistemic =  np.mean(-np.log10(epistemic_all+ 1e-12))
            epistemics.append(epistemic )
            # print(f"Orignial Epistemic uncertainty: {epistemic}")

            # 2. Call IForest
            # contamination = list(label).count(1)/max_length
            contamination = 0.1
            model_functions = {
            'IForest': lambda: IF_o(df, X_data, label, slidingWindow, contamination, random_state = random_state),
            'SAND': lambda: sand_o(df,  data, label, slidingWindow, contamination),
            'LOF': lambda: lof_o(df,  X_data, label, 
                                slidingWindow, contamination=contamination),
            'DAMP': lambda: damp_o(df,  data, label, slidingWindow, contamination),
            'MatrixProfile': lambda:  matrixProfile_o(df,  data, label, slidingWindow, contamination),
            'Ser2Graph': lambda:  Ser2Graph_o(df,  data, label, slidingWindow, contamination),
            'AutoEncoder': lambda:  AE_o(df,  data, label, slidingWindow, contamination),
                            }
            if model in model_functions:
                ROC, PR, F, execution_time = model_functions[model]()
            else:
                raise ValueError(f"Unsupported model: {model}")
            log(f"{index+1}.{filename[:18]:<18}:  AUC_ROC = {ROC:<10.5f} AUC_PR = {PR:<10.5f} "\
                  f"F = {F:<10.5f} Time = {execution_time:<3.2f}S")
            ROC_mean += ROC
            PR_mean += PR
            F_mean += F
            Time_mean += execution_time 
            n += 1

        log(f"Average AUC_ROC :  {ROC_mean/n}")
        log(f"Average AUC_PR :   {PR_mean/n}")
        log(f"Average F :   {F_mean/n}")
        log(f"Average Time :   {Time_mean/n}S\n")
        log(f"epistemics :   {epistemics}\n")


if __name__ =="__main__":
    modelName=['IForest', 'LOF', 'SAND']
    pathList = [
                # "/home/guoyou/OutlierDetection/TSB-UAD/data/SED/",
                # "/home/guoyou/OutlierDetection/TSB-UAD/data/Genesis/" ,
                "/home/guoyou/OutlierDetection/TSB-UAD/data/ECG/",
                # "/home/guoyou/OutlierDetection/TSB-UAD/data/MGAB/",
                # "/home/guoyou/OutlierDetection/TSB-UAD/data/SensorScope/",
                # "/home/guoyou/OutlierDetection/TSB-UAD/data/MITDB/",
                # "/home/guoyou/OutlierDetection/TSB-UAD/data/Daphnet/",
                # "/home/guoyou/OutlierDetection/TSB-UAD/data/IOPS/",
                #  "/home/guoyou/OutlierDetection/TSB-UAD/data/synthetic/",

                #  "/home/guoyou/OutlierDetection/TSB-UAD/data/YAHOO/",
                # "/home/guoyou/OutlierDetection/TSB-UAD/data/Dodgers/",
                # "/home/guoyou/OutlierDetection/TSB-UAD/data/KDD21/",
               ]
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_state", type=int, default=1024, help="Random seed")
    args = parser.parse_args()
    random_state = args.random_state
    random.seed(20250412)
    random_state = random.randint(1, 4294967295)
    with open("OutlierDetection/TSB-UAD/example/results_Orig.txt", "w") as output_file2:
        for i in [0,1,2]:
            run(pathList, model=modelName[i], output_file=output_file2, random_state=random_state)
