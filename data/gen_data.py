import json
from pprint import pprint
import collections
from collections import Counter
import operator
import os



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

print("\nConverting labels for a start to normal/abnormal")
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


print("\nSplitting data : 0.6 train, 0.2 validation and 0.2 for test!")

d = {int(k):v for k,v in new_dict.items()}
od = collections.OrderedDict(sorted(d.items()))
id_im = 1
labels = list(od.values())

labels_train = labels[:int(len(labels)*0.6)]
labels_val = labels[int(len(labels)*0.6):int(len(labels)*0.8)]
labels_test = labels[int(len(labels)*0.8):]

print("\n\tsize of train data is",len(labels_train))
print("\tsize of val data is",len(labels_val))
print("\tsize of test data is",len(labels_test))

print("\nGenerating text files of data !")

f = open('train.txt','w')
for lab in labels_train :
    f.write(lab+" "+os.path.abspath("images/"+str(id_im)+".png\n"))
    id_im +=1
f.close()

f = open('val.txt','w')
for lab in labels_val :
    f.write(lab+" "+os.path.abspath("images/"+str(id_im)+".png\n"))
    id_im +=1
f.close()

f = open('test.txt','w')
for lab in labels_test :
    f.write(lab+" "+os.path.abspath("images/"+str(id_im)+".png\n"))
    id_im +=1
f.close()

print("\nTxt files generated succefully, Happy training !")
