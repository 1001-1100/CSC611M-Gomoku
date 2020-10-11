import requests
import json

n = 100
m = 255
r = requests.get('https://api-randomizer.herokuapp.com/integer?n='+str(n)+'&m='+str(m))
integers = json.loads(r.text)