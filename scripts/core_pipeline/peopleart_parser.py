#!/usr/bin/env python3
import xml.etree.ElementTree as ET
from pathlib import Path

def parse_peopleart_xml(xml_path):
    """
    Parses a PeopleArt PASCAL VOC XML file to extract ground-truth boxes for 'person'.
    Returns a tuple: (width, height, list of boxes)
    Each box is a dict: {'name': 'person', 'xmin': float, 'ymin': float, 'xmax': float, 'ymax': float}
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get image dimensions
        size_node = root.find('size')
        if size_node is not None:
            width = int(size_node.find('width').text)
            height = int(size_node.find('height').text)
        else:
            return None, None, []
            
        boxes = []
        for obj in root.findall('object'):
            name = obj.find('name').text.lower().strip()
            if name == 'person':
                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                
                boxes.append({
                    'name': 'person',
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax
                })
        return width, height, boxes
    except Exception as e:
        print(f"Error parsing XML {xml_path}: {e}")
        return None, None, []

if __name__ == '__main__':
    # Quick test
    sample_xml = Path("/data/brhanu/datasets/PeopleArt-master/Annotations/cartoon/011.jpg.xml")
    if sample_xml.exists():
        w, h, boxes = parse_peopleart_xml(sample_xml)
        print(f"Parsed {sample_xml.name}: Size {w}x{h}, Found {len(boxes)} persons:")
        for idx, box in enumerate(boxes, 1):
            print(f"  Box {idx}: xmin={box['xmin']}, ymin={box['ymin']}, xmax={box['xmax']}, ymax={box['ymax']}")
    else:
        print("Sample XML not found.")
