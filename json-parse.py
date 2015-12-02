import os
from pprint import pprint
import simplejson as json

for (dirpath, dirname, filenames) in os.walk('results'):
    for fname in filenames:
        fpath = os.path.join(dirpath, fname)
        with open(fpath) as f:
        	j = json.load(f, encoding="latin-1")
        	print json.dumps(j, indent=4*" ")
