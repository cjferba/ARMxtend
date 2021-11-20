import pickle
import itertools


def cargar_datos(path="/home/carlos/Gits/Gitlab/AR_SPARK/Result/Angel Rules/0.003Items.dat"):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (OSError, IOError) as e:
        return dict()


def guardar_datos(dic):
    with open("Items.dat", "wb") as f:
        pickle.dump(dic, f)


setOfRules = []
LisOfDict = cargar_datos("/home/carlos/Gits/Gitlab/AR_SPARK/0.001Items.dat")

d = LisOfDict[len(LisOfDict)-1]

for key, value in d.items():
    keys = key.split("-")
    listapermu=[]
    for i in range(1,len(keys)-1):
       # print(keys)
        listapermu=listapermu+(list(itertools.combinations(keys,i)))
   # print(listapermu)
    Ban=True
    listapermu = list(map('-'.join, listapermu))
    for p in range(0, len(listapermu)):
        if Ban != True:
            pcom = listapermu[p].split("-")
            for k in keys:
                if listapermu[p] != k and not (k in pcom):
                    id = len(listapermu[p].split("-")) - 1
                    keysearch = listapermu[p]
                    Rule1 = [(listapermu[p]) + "-->" + (k), value / LisOfDict[id][keysearch]]
                    id = len(k.split("-")) - 1
                    keysearch = k
                    Rule2 = [(k) + "-->" + (listapermu[p]), value / LisOfDict[id][keysearch]]
        else:
            pcom = []
            for k in range(1, len(keys)):
                if listapermu[p] != keys[k] and not (keys[k] in pcom):
                    id = len(listapermu[p].split("-")) - 1
                    keysearch = listapermu[p]
                 #   print(LisOfDict[id][keysearch])
                    Rule1 = [(listapermu[p]) + "-->" + (keys[k]), value / LisOfDict[id][keysearch]]
                    id = len(keys[k].split("-")) - 1
                    keysearch = keys[k]
                  #  print(LisOfDict[id][keysearch])
                    Rule2 = [(keys[k]) + "-->" + (listapermu[p]), value / LisOfDict[id][keysearch]]
        setOfRules.append(Rule1)
        setOfRules.append(Rule2)

print(setOfRules)
# for j in range(0,len(keys)):
# newstr = oldstr.replace("M", "")
