



#items2=GenItems(2, items)
#items3=GenItems(3, items2)
items3=['A-B-C', 'A-B-D', 'A-C-D', 'A-C-B', 'A-D-B', 'A-D-C', 'B-C-D', 'B-D-C']
items4=GenItems(3, items3)
print(items4)


# A = {10, 20, 30, 40, 80}
# B = {100, 30, 80, 40, 60}
# print (A.difference(B))
# print (B.difference(A))
#
sets=set(frozenset(x.split("-")) for x in items3)
#
# sets.issubset(set(frozenset(['A','B','C'])))


items=['A', 'B','C' , 'D']
items3=['A-B-C', 'A-B-D', 'A-C-D', 'A-C-B', 'A-D-B', 'A-D-C', 'B-C-D', 'B-D-C']
DIC={}
DIC[frozenset(['A','B','C'])]=0.3
fitems=frozenset(items)
sets=set(frozenset(x.split("-")) for x in items3)

sets.intersection(set(frozenset(['C' , 'D', 'A'])))