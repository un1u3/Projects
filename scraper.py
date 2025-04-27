import requests
from bs4 import BeautifulSoup

URL = 'https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html'
headers = {"User-Agent":'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36'}


page = requests.get(URL, headers=headers)
soup = BeautifulSoup(page.content, 'html.parser')


# print(soup.prettify())

title = soup.find(id='productTitle')
print(title)

# print(title.strip())s
