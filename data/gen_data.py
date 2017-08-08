import json
from pprint import pprint
import collections
from collections import Counter
import operator
import os
import logging
import sys


logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


dic = {}
with open('data_new.json') as data_file:
    data = json.load(data_file)
    lst = []
    n=0
    for  i in data:
        dic[i] = "/".join(data[i]['items'])

val_counter = Counter(dic.values())

sorted_x = list(reversed(sorted(val_counter.items(), key=operator.itemgetter(1))))
logging.info("The classes with most occurence in the dataset:\n")
for ob in sorted_x[:10] :
    print("\t\t",ob)
print("\n")
logging.info("Total number of Images : {}".format(len(dic)))
logging.info("number of normal chest Images(healthy people) {}:".format(sorted_x[0][1]))
logging.info("number of abnormal chest Images(sick people) {}:".format(len(dic)-sorted_x[0][1]))

logging.info("Converting labels for a start to normal/abnormal")
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

d = {int(k):v for k,v in new_dict.items()}
od = collections.OrderedDict(sorted(d.items()))
id_im = 1
labels = list(od.values())

logging.info("size of data is {}".format(len(labels)))

logging.info("Generating text files of data !")

f = open('data.txt','w')
for lab in labels :
    if os.path.isfile("images/"+str(id_im)+".png"):
        f.write(lab+" "+os.path.abspath("images/"+str(id_im)+".png\n"))
    id_im +=1
f.close()

logging.info("Txt files generated succefully, Happy training !")
