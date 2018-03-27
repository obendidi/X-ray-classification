from lxml import html
import requests
import re
import json
import urllib
import sys
import os
import logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


path = "images/"

logging.info("Saving images in {}".format(path))
if not os.path.isdir(path):
    os.mkdir(path)
else :
    logging.info("Found path already exsting , removing all images to start again !")
    files = os.listdir(path)
    for f in files:
        os.remove(path+f)


domain = 'https://openi.nlm.nih.gov/'
url_list = []
for i in range(0,75):
    url = 'https://openi.nlm.nih.gov/gridquery.php?q=&it=x,xg&sub=x&m='+str(1+100*i)+'&n='+str(100+100*i)
    url_list.append(url)
regex = re.compile(r"var oi = (.*);")
final_data = {}
img_no = 0


def extract(url):
    global img_no

    try :
        img_no += 1
        r = requests.get(url)
        tree = html.fromstring(r.text)

        div = tree.xpath('//table[@class="masterresultstable"]\
            //div[@class="meshtext-wrapper-left"]')
    except : div=[]

    if div != []:
        div = div[0]
    else:
        return

    typ = div.xpath('.//strong/text()')[0]
    items = div.xpath('.//li/text()')
    img = tree.xpath('//img[@id="theImage"]/@src')[0]


    final_data[img_no] = {}
    final_data[img_no]['type'] = typ
    final_data[img_no]['items'] = items
    final_data[img_no]['img'] = domain + img
    try :
        urllib.request.urlretrieve(domain+img, path+str(img_no)+".png")
        with open('data_new.json', 'w') as f:
            json.dump(final_data, f)

        output = "Downloading Images : {}".format(img_no)
        sys.stdout.write("\r\x1b[K" + output)
        sys.stdout.flush()
    except :return


def main():
    for url in url_list :
        try:
            r = requests.get(url)
        except : continue
        tree = html.fromstring(r.text)

        script = tree.xpath('//script[@language="javascript"]/text()')[0]

        json_string = regex.findall(script)[0]
        json_data = json.loads(json_string)

        next_page_url = tree.xpath('//footer/a/@href')

        links = [domain + x['nodeRef'] for x in json_data]
        for link in links:
            extract(link)

if __name__ == '__main__':

    main()


#python scraper.py <path to folders>
