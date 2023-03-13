import urllib.parse
import json
import requests
# import jsonpath
# from tqdm import tqdm
#
# # url = 'https://www.duitang.com/napi/blog/list/by_search/?kw={}&start={}'
# url2='https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&sf=1&fmq=&pv=&ic=0&nc=1&z=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&fm=index&pos=history&word=%E8%BD%A6%E7%A5%B8'
#
# # label = '车祸'
# # label = urllib.parse.quote(label)
#
# num = 0
# for index in range(0,2400,24):
#     # u = url.format(label,index)
#     u=url2
#     we_data = requests.get(u).text
#     html = json.loads(we_data)
#     photo = jsonpath.jsonpath(html,"$..path")
#
#     if photo:
#         for i in tqdm(photo):
#             a = requests.get(i)
#             with open(r'D:\毕业\毕设\project\weibo-search\结果文件\image3\车祸\{}.jpg'.format(num),'wb') as f:
#                 f.write(a.content)  # 二进制
#             num += 1


import re
import os
import requests
import tqdm

header = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Safari/537.36'}


def getImg(url, idx, path):
    img = requests.get(url, headers=header)
    file = open(path + str(idx) + '.jpg', 'wb')
    file.write(img.content)
    file.close()


search = input("请输入搜索内容：")
number = int(input("请输入需求数量："))
path = './结果文件/image3/' + search + '/'
if not os.path.exists(path):
    os.makedirs(path)

bar = tqdm.tqdm(total=number)
page = 0
while (True):
    if number == 0:
        break
    url = 'https://image.baidu.com/search/acjson'
    params = {
        "tn": "resultjson_com",
        "logid": "11555092689241190059",
        "ipn": "rj",
        "ct": "201326592",
        "is": "",
        "fp": "result",
        "queryWord": search,
        "cl": "2",
        "lm": "-1",
        "ie": "utf-8",
        "oe": "utf-8",
        "adpicid": "",
        "st": "-1",
        "z": "",
        "ic": "0",
        "hd": "",
        "latest": "",
        "copyright": "",
        "word": search,
        "s": "",
        "se": "",
        "tab": "",
        "width": "",
        "height": "",
        "face": "0",
        "istype": "2",
        "qc": "",
        "nc": "1",
        "fr": "",
        "expermode": "",
        "force": "",
        "pn": str(60 * page),
        "rn": number,
        "gsm": "1e",
        "1617626956685": ""
    }
    result = requests.get(url, headers=header, params=params).json()
    url_list = []
    for data in result['data'][:-1]:
        url_list.append(data['thumbURL'])
    for i in range(len(url_list)):
        getImg(url_list[i], 60 * page + i, path)
        bar.update(1)
        number -= 1
        if number == 0:
            break
    page += 1
print("\nfinish!")
