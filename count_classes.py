import xml.etree.ElementTree as ET
from pathlib import Path

source_dir = "/home/tanaka/Projects/crawl_profile_cards/labels/"

result = {}
for path in Path(source_dir).glob("*.xml"):
    tree = ET.parse(str(path))
    root = tree.getroot()
    objitr = root.getiterator("object")
    for obj in objitr:
        clazz = obj.findtext(".//name")
        result[clazz] = 1 if clazz not in result else result[clazz] + 1

print(result)
        
    
