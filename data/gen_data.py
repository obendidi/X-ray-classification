import json
from pprint import pprint
from collections import Counter
import operator




dic = {}
with open('data_new.json') as data_file:
    data = json.load(data_file)
    lst = []
    n=0
    for  i in data:
        # print i
        dic[i] = "/".join(data[i]['items'])

val_counter = Counter(dic.values())

sorted_x = list(reversed(sorted(val_counter.items(), key=operator.itemgetter(1))))
print("\nThe classes with most occurence in the dataset:\n")
for ob in sorted_x[:10] :
    print("\t",ob)
print("\nTotal number of Images :",len(dic))
print("\nnumber of normal chest Images(healthy people) :",sorted_x[0][1])
print("\nnumber of abnormal chest Images(sick people) :",len(dic)-sorted_x[0][1])

print("Converting labels for a start to normal/abnormal")
st=""
new_dict={}
check = ["normal"]
for j in dic:
    for i in check:

        if i in dic[j].lower():
            st+=i
    if st == "":
        st += "abnormal"
    new_dict[j] = st

    st=""
