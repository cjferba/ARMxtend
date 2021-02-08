import pandas as pd
import numpy as np

class FuzzyLib():
    def __init__(self):
        self.data = pd.DataFrame()
        self.atributes=[]
        self.types = []
    def LoadData(self,File=""):
        if File=="":
            print("Error File")
        else:
            self.data=pd.read_csv(File)
            self.atributes = self.data.columns
            self.types = self.data.dtypes

    def GetData(self):
        return self.data
    def GetAtributes(self):
        return self.atributes
    def GetTypes(self):
        return self.types

    def Fuzzification(self,Atri=[],thresholds=[], FuzzyLabel=[]):
        def function(data):
            #df["range"] = pd.cut(df['val'], ranges)
            return data
        for i in range(0,len(Atri)):
            # self.data[i]=list(map(function, self.data[i]))
            print(self.types[i])
            print(self.data[Atri[i]])
            if self.types[i]=="float":
                print(self.data[Atri[i]])

                self.data[Atri[i]].quantile(.5)
        return 0


df=pd.read_csv("../FuzzylibEnergytest/Interpolate_ICPE_2016-01-07_2015-05-30_Wetaher.csv", low_memory=False)
# df.set_index("date")
# df = df.drop("date.1",1)

SensorList=['date','Temperature','Humidity','Wind Speed','Pressure','Precip.','occupation',
            'dayofWeek','8877','8883','8901','8918','8929','8930','8933','8937','8951','8957',
            '8963','8964','8971','8974','8975','8978','8982','8985','9015','9013','9020','9021',
            '9028','9029','9032','9038','9039','9040','9041','9042','9043','9044','9045','9046',
            '9048','9047','9049','9050','9051','9052','9053','9054','9055','9056','9058','9059',
            '9060','9061','9062','9063','9064','9065','9066','9067','9068','9069','9070','9071',
            '9075','9077','9078','9079','9080','9081','9082','9083','9084','9085','9086','9087',
            '9088','9089','9092','9094','9096','9097','9098','13552','13553','13554']

df=df[SensorList]
df=df.set_index("date")
SensorList=['Temperature','Humidity','Wind Speed','Pressure','Precip.','occupation',
            'dayofWeek','8877','8883','8901','8918','8929','8930','8933','8937','8951','8957',
            '8963','8964','8971','8974','8975','8978','8982','8985','9015','9013','9020','9021',
            '9028','9029','9032','9038','9039','9040','9041','9042','9043','9044','9045','9046',
            '9048','9047','9049','9050','9051','9052','9053','9054','9055','9056','9058','9059',
            '9060','9061','9062','9063','9064','9065','9066','9067','9068','9069','9070','9071',
            '9075','9077','9078','9079','9080','9081','9082','9083','9084','9085','9086','9087',
            '9088','9089','9092','9094','9096','9097','9098','13552','13553','13554']
# 'Humidity','Wind Speed','Pressure','Precip.','occupation',
Intervals={
'Temperature':[[18,20,24,26],["cold","conford","warm"]],
'dayofWeek':[[0,1,2,3,4,5,6],["Mon","Tu","Wed","Thu","Fri","Sat","Sun"]]
}

for i in range(0,len(SensorList)):
    print(SensorList[i])
    # print((df[SensorList[i]].unique()))
    if len(np.unique(df[SensorList[i]][~np.isnan(df[SensorList[i]])]))==1:
        print("------------------------------"+"Drop "+str(SensorList[i])+"------------------------------")
        df = df.drop(SensorList[i], axis=1)
    elif set(np.unique(df[SensorList[i]][~np.isnan(df[SensorList[i]])])).issubset(set([0,1])):
        mask = (df[SensorList[i]] == 1)
        df[str(SensorList[i]) + "_on"] =0
        df[str(SensorList[i]) + "_off"] = 0
        df[str(SensorList[i])+"_on"][mask]=1
        df[str(SensorList[i]) + "_off"][~mask] = 1
        # df[SensorList[i]]=pd.Categorical(df[SensorList[i]])
        print("########################\t ON/OFF \t######################")
    # elif len(Values[i])!=0:
    #     print("Drop " + str(SensorList[i]))
    else:
        mask = (df[SensorList[i]] >= df[SensorList[i]].quantile(0.25)) & (df[SensorList[i]] <= df[SensorList[i]].quantile(0.25))
        FullData = df.loc[mask]
        print("low de "+str(df[SensorList[i]].quantile(0.001))+" a "+str(df[SensorList[i]].quantile(0.25))+
              " moderate a"+str(df[SensorList[i]].quantile(0.375))+
              "high de"+str(df[SensorList[i]].quantile(0.75))+" a "+str(df[SensorList[i]].quantile(0.875)))

