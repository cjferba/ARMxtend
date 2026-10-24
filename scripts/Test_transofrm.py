import pandas as pd
Data = pd.read_csv("ARMxtend/dataset/Interpolate_ICPE_2016-01-07_2015-05-30_Wetaher.csv");

Names=['Temperature', 'Dew Point', 'Humidity', 'Wind', 'Wind Speed',
       'Wind Gust', 'Pressure', 'Precip.', 'Condition', 'occupation',
       'dayofWeek', '8877', '8883', '8901', '8918', '8929', '8930', '8933',
       '8937', '8951', '8957', '8963', '8964', '8971', '8974', '8975', '8978',
       '8982', '8985', '9015', '9013', '9020', '9021', '9028', '9029', '9032',
       '9038', '9039', '9040', '9041', '9042', '9043', '9044', '9045', '9046',
       '9048', '9047', '9049', '9050', '9051', '9052', '9053', '9054', '9055',
       '9056', '9058', '9059', '9060', '9061', '9062', '9063', '9064', '9065',
       '9066', '9067', '9068', '9069', '9070', '9071', '9075', '9077', '9078',
       '9079', '9080', '9081', '9082', '9083', '9084', '9085', '9086', '9087',
       '9088', '9089', '9092', '9094', '9096', '9097', '9098', '13552',
       '13553', '13554']

Data=Data.drop(['9094', '9051', '9052','9029','9013','Precip.'], axis=1)
Data["date"]=pd.to_datetime(Data["date"])
Names = Data.select_dtypes(include=['float']).columns


DataNew=pd.DataFrame()
for i in Names:
    print(i)
    q=Data[i].quantile([0,.25, .5,.75,1]).unique()

    # if (len(q)==1):
    #     print(str(i) + "   " + str(Data[i].unique()))
    # else:
    #     print(str(i)+"   "+str(q))
    if (len(q)>4):
        print(str(i) + "   3--------" + str(q))
        # DataNew[i] = pd.qcut(Data[i], q=3)
        DataNew[i] = pd.cut(Data[i], bins=3)

    elif(len(q)>=2):
        print(str(i) + "   2-----" + str(q))
        # DataNew[i] = pd.qcut(Data[i], q=2)
        DataNew[i] = pd.cut(Data[i], bins=2)
    else:
        print("Error" + str(i))
        #Data.drop([i])
import csv

DataNew=pd.DataFrame({col:str(col)+'=' for col in DataNew}, index=DataNew.index) + DataNew.astype(str)
DataNew.to_csv("ARMxtend/dataset/Interpolate_ICPE_2016-01-07_2015-05-30_Wetaher_Discrete.csv",index=False, sep=";",quoting=csv.QUOTE_NONE)
