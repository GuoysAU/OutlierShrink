import argparse
from Package import *




def run(pathList, model, output_file, random_state=None, snr = 25):  
    def log(message, end="\n"):
        print(message, end=end, flush=True)  
        output_file.write(message + end) 
        output_file.flush()  
    log("==================================================")
    log(f"Starting analysis for Model: {model}")
    log("==================================================") 
    filenamesList = [Filenames2List(path) for path in pathList]
    # filenamesList = filenamesList[-11:-10]
    for i, f in enumerate(filenamesList):
        path = pathList[i]
        ROC_mean, PR_mean, F_mean, Time_mean, n = 0, 0, 0, 0, 0
        # log(f"Dataset: {path.split('data')[1].split('/')[1]} after compression", end="")
        log(f"Dataset: {path.split('data')[1].split('/')[1]} after compression")
        index = 0
        epistemics = []
        for index, filename in enumerate(f):
            # 1. Compress and transform data
            # snr = 20
            snr = estimate_snr_from_file(path+filename,snr_min=25)

            df = pd.read_csv(path+filename, header=None).dropna()
            
            # #************************************************************
            # max_length = df.shape[0]
            # data = df.iloc[:max_length, 0].astype(float)
            # WindowSize = find_length(data)
            # #************************************************************

            label = df.iloc[:, 1].astype(int)
            if((list(label).count(1))==0):
                log(f"{index+1}.{filename[:15]:<15}: No Anomoly")
                continue
            df_comp = compress(path, filename, snr=snr)
            # if(df_comp.shape[0]/df.shape[0]<0.01): #要使得点的压缩情况不能小于1%，例如SED
            #     print(f"\t Low number of points, Less than 0.01, current snr = {snr}")
            #     snr += 5
            #     df_comp  = compress(path, filename, snr)
            #     log(" "+f"{filename}:  snr = {snr}")
            # elif(index<1):
            #     log(" ")

            

            # 2. Prepare the compressed data for unsupervised method
            df_selected = df.loc[df_comp['index']]  
            df_selected[0] = df_comp['value'].values
            df_selected = df_selected.to_numpy()
            data_selected = df_selected[:,0].astype(float)
            label_selected = df_selected[:,1].astype(int)
            slidingWindow_selected = find_length(data_selected)


            #****************************************#
            if(model=="LOF"):
                slidingWindow_selected  = 1
            #****************************************#

             #****************************************#
            if(model=="IForest"):
                slidingWindow_selected  = 1
            #****************************************#


            X_data_selected = Window(window = slidingWindow_selected).convert(data_selected).to_numpy()

            # # 3. Data characteristics
            # print("Time series length (Original): ", df.shape[0])
            # print("Time series length (Compressed): ",len(data_selected))
            # print("Estimated Subsequence length (Compressed): ",slidingWindow_selected)
            # print("Number of abnormal points (Compressed): ",list(label_selected).count(1))
            # abnormal_indices = np.where(label_selected == 1)[0]
            # # print("Index of anomoly: \n", abnormal_indices)                              # Relative Position of index
            # print("Index of anomoly: \n", df_comp.iloc[:, 0][abnormal_indices].to_numpy()) # Actual index
            # # if(len(abnormal_indices)!=0):
            #     abnormal_indices = np.where(label_selected == 1)[0]
            #     abnormal_points = [[idx, data_selected[idx]] for idx in np.where(label_selected == 1)[0]]
            #     print("Pair of Anomoly: ")
            #     for i in range(0, len(abnormal_points), 12):
            #         formatted_row = ["[{:<6}, {:>10.4f}]".format(idx, val) for idx, val in abnormal_points[i:i+12]]
            #     print(" ".join(formatted_row))
            # else:
            #     raise Exception("Sorry, no amomoly found after compression")


            # # 4. Figure to show
            # df_temp = pd.read_csv(path+filename, header=None).dropna()
            # start, end = 0, 7000
            # values_temp = df_temp[0].iloc[start:end]
            # labels_temp = df_temp[1].iloc[start:end]

            # comp_filtered = df_comp[(df_comp['index'] >= start) & (df_comp['index'] < end)]
            # comp_index = comp_filtered['index']
            # comp_values = comp_filtered['value']

            # plt.figure(figsize=(12, 8))
            # plt.scatter(values_temp[labels_temp == 0].index, values_temp[labels_temp == 0], color='blue', label='Normal (Original)', s=10)
            # plt.scatter(values_temp[labels_temp == 1].index, values_temp[labels_temp == 1], color='red', label='Outlier (Original)', s=10)
            # plt.scatter(comp_index, comp_values, color='green', label='Sampled Data (df_comp)', s=20, marker='x')

            # plt.title(f"Original Data with Tranformed Data Highlighted (Index {start}-{end})")
            # plt.xlabel("Index")
            # plt.ylabel("Value")
            # plt.legend()
            # plt.show()

            # 5. Call IForest
            # contamination = list(label_selected).count(1)/len(data_selected)
            # print(f"Actual contamination = {list(label_selected).count(1)/len(data_selected)}")
            # contamination = np.nextafter(0, 1) if contamination == 0. else contamination
            # contamination = 0.1 if contamination == 0. else contamination

            clf = IForest(n_jobs=1, contamination=0.1, random_state=random_state)
            clf.fit(X_data_selected)
            epistemic = clf.epistemic_uncertainty
            epistemic_all = np.zeros(df.shape[0] )
            epistemic_all[:epistemic.shape[0]] = epistemic
            epistemic =  np.mean(-np.log10(epistemic_all+ 1e-12))
            epistemics.append(epistemic )
            # print(f"Compreessed Epistemic uncertainty: {epistemic}")



            contamination = 0.1
            model_functions = {
            'IForest': lambda: IF(df, df_comp, X_data_selected, label_selected, 
                                slidingWindow_selected, contamination=contamination, random_state=random_state),
            'SAND': lambda: sand(df, df_comp, data_selected, label_selected, 
                                slidingWindow_selected, contamination=contamination, random_state=random_state),
            'LOF': lambda: lof(df, df_comp, X_data_selected, label_selected, 
                                slidingWindow_selected, contamination=contamination, random_state=random_state),
                            }
            # Call the appropriate function or raise an error if the model is not supported
            if model in model_functions:
                ROC, PR, F, execution_time = model_functions[model]()
            else:
                raise ValueError(f"Unsupported model: {model}")
            log(f"{index+1}.{filename[:]:<30}:  AUC_ROC = {ROC:<10.5f} AUC_PR = {PR:<10.5f} "\
                  f"F = {F:<10.5f} Time = {execution_time:<3.5f}S")
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
    with open("OutlierDetection/TSB-UAD/example/results__Comp.txt", "w") as output_file:
        for i in [0,1,2]:
            run(pathList, model=modelName[i], output_file=output_file, random_state=random_state)






























