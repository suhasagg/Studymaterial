import scrapy
from bs4 import BeautifulSoup

class WittyfeedSpider(scrapy.Spider):
   name = 'wittyfeed'
   with open('/root/wittyfeedurl1.txt') as f:
       start_urls = [url.strip() for url in f.readlines()]
  

    
   

   def parse(self, response):
    soup = BeautifulSoup(response.text, "html.parser")
    d= {}
    data={}
    global k
    publisher= ""
    author=""
    section=""
    title=""
    publishedTime=""
    i=1
    additiontime=""
    tag=""
    scraped_info={}
    a=""
    b=""
    c=""
    d=""
    e=""
    f=""
    g=""
    publisher = soup.find("meta", property="article:publisher")
    a=publisher['content']
    author = soup.find("meta", property="article:author")
    b=author['content']
    section = soup.find("meta", property="article:section")
    c=section['content']
    title = soup.find("meta", property="og:title")
    d=title['content']
    publishedTime = soup.find("meta", property="article:published_time")
    e=publishedTime['content']
    image = soup.find("meta", property="og:image")
    f=image['content']
    tag = soup.findAll("meta", property="article:tag")
    for tagdata in tag:
        g=tagdata['content']+","
    g=g[:-1]
    


    print(a)
    print(b)
    print(c)
    print(d)
    print(e)
    print(f)
    print(g)
    import datetime

    additiontime= datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    from urllib.parse import urlparse 
    parsedurl = urlparse(response.url)

    import hashlib
    pbHash=hashlib.sha1(str(response.url).encode('utf-8')).hexdigest()

    d["Entity6"]=g.strip().replace(",",";")
    d["Entity4"]=b.strip().replace(",",";")
    d["Entity10"]=""
    d["Entity7"]=f
    d["Entity5"]=e.strip().replace("T"," ")
    d["Entity3"]=""
    d["Entity6"]=""
    d["Entity11"]="cafc82b3ceae411d8bc38bced7a1fff3"
    d["Entity1"]=response.url
    d["Entity12"]=c.strip().replace(",",";")
    d["Entity9"]=d.strip().replace(",",";")
    d["Entity2"]=parsedurl.path
    d["Entity13"]=additiontime
    d["EntityId"]=pbHash
    d["Entity8"]=""
 #   yield d.values()

    import json
    #with open('data30.json', 'a') as fp:
    import codecs
    fp= codecs.open('wittyfeeditems.json', mode = 'a', encoding = 'utf-8')
    line1 = '{"index":{}}'+"\n"
    line = line1 + json.dumps(d) + "\n"
    fp.write(line)
    fp.close
