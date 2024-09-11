import argparse
import os
import json
import xml.etree.ElementTree as ET
import cv2

class AnnotationConverter:
    def __init__(self,input_file,output_dir):
        self.input_file=input_file
        self.output_dir=output_dir
        self.data=None

    def load_data(self):
        raise NotImplementedError("This method should be implemented by subclasses")
    
    def convert_to(self,target_format,**kwargs):
        raise NotImplementedError("This method should be implemented by subclasses")

    def save_data(self,output_file):
        raise NotImplementedError("This method should be implemented by subclasses")


## COCO Converter class

class COCOConverter(AnnotationConverter):
    def load_data(self):
        with open(self.input_file,'r')as f:
            self.data=json.load(f)

    def convert_to(self, target_format, **kwargs):
        if target_format=="VOC":
            return self.convert_to_voc()
        elif target_format=="YOLO":
            return self.convert_to_yolo()
        else:
            raise ValueError(f"Unsupported target format: {target_format}")
        
    def convert_to_voc(self):
        annotations=[]
        for image in self.data['images']:
            annotation=ET.Element('annotation')
            ET.SubElement(annotation,"folder").text=self.output_dir
            ET.SubElement(annotation,"filename").text=image['file_name']

            size=ET.SubElement(annotation,"size")
            ET.SubElement(size,'width').text=str(image['width'])
            ET.SubElement(size,"height").text=str(image['height'])
            ET.SubElement(size,'depth').text="3"

            for ann in self.data['annotations']:
                if ann['image_id']==image['id']:
                    obj=ET.SubElement(annotation,"object")
                    category_name=[nm['name'] for nm in self.data['categories'] if nm['id']==ann['category_id']][0]

                    bbox=ann['bbox']
                    xmin=int(bbox[0])
                    ymin=int(bbox[1])
                    xmax=int(bbox[0]+bbox[2])
                    ymax=int(bbox[1]+bbox[3])

                    ET.SubElement(obj,"name").text=category_name
                    ET.SubElement(obj,"pose").text="Unspecified"
                    ET.SubElement(obj,"truncated").text='0'
                    ET.SubElement(obj,"difficult").text='0'

                    bndbox=ET.SubElement(obj,"bndbox")
                    ET.SubElement(bndbox,"xmin").text=str(xmin)
                    ET.SubElement(bndbox,"ymin").text=str(ymin)
                    ET.SubElement(bndbox,"xmax").text=str(xmax)
                    ET.SubElement(bndbox,"ymax").text=str(ymax)

                    if 'segmentation' in ann and ann['segmentation']:
                        polygon = ann['segmentation'][0]
                        polygon_points = ' '.join([f'{x},{y}' for x, y in zip(polygon[0::2], polygon[1::2])])
                        ET.SubElement(obj, "polygon").text = polygon_points
            annotations.append(ET.ElementTree(annotation))
        return annotations
    
    def convert_to_yolo(self):
        yolo_annotations=[]
        class_to_index={nm['name']:idx for idx,nm in enumerate(self.data['categories'])}
        for ann in self.data['annotations']:
            image=next(img for img in self.data['images'] if img['id']==ann['image_id'])
            class_index=class_to_index[[nm['name'] for nm in self.data['categories'] if nm['id']==ann["category_id"]][0]]
            bbox=ann['bbox']
            x_center=(bbox[0]+bbox[2]/2)/image['width']
            y_center=(bbox[1]+bbox[3]/2)/image['height']
            width=bbox[2]/image['width']
            height=bbox[3]/image['height']
            yolo_annotation=f"{class_index} {x_center:.3f} {y_center:.3f} {width:.3f} {height:.3f}"
            yolo_annotations.append((image['file_name'],yolo_annotation))
       
        return yolo_annotations
    
    def save_data(self,annotations,target_format):
        if target_format=="VOC":
            for annotation in annotations:
                xml_output_path=os.path.join(self.output_dir,os.path.splitext(annotation.find('filename').text)[0]+".xml")
                annotation.write(xml_output_path)
        elif target_format=="YOLO":
            annotations_by_image = {}
            for filename, yolo_annotation in annotations:
                if filename not in annotations_by_image:
                    annotations_by_image[filename] = []
                annotations_by_image[filename].append(yolo_annotation)
           
            for filename, yolo_annotations in annotations_by_image.items():
                yolo_output_path = os.path.join(self.output_dir, os.path.splitext(filename)[0] + '.txt')
                with open(yolo_output_path, 'w') as f:
                    for annotation in yolo_annotations:
                        f.write(annotation + '\n')


class PASCALVOCCOnverter(AnnotationConverter):
    def __init__(self, input_file, output_dir, starting_image_id=1, starting_annotation_id=1):
        super().__init__(input_file, output_dir)
        self.image_id = starting_image_id
        self.annotation_id = starting_annotation_id
        self.image_data = []
        self.annotation_data = []
        self.category_map = {}
        self.output_file = os.path.join(self.output_dir, 'coco_annotations.json')

    def load_data(self):
        tree=ET.parse(self.input_file)
        root=tree.getroot()
        self.data=root
        print(self.data)
    
    def convert_to(self,target_format,**kwargs):
        if target_format=='COCO':
            return self.convert_to_coco()
        elif target_format=="YOLO":
            return self.convert_to_yolo()
        else:
            return ValueError(f"Unsupported format {target_format}")
    
    def convert_to_coco(self):
    
        if not os.path.isfile(self.output_file):
            coco_data = {
                "info": {
                    "year": 2024,
                    "version": "1.0",
                    "description": "Converted COCO dataset",
                    "contributor": "Himanshi Sharma",
                    "url": "http://example.com",
                    "date_created": "2024-09-01"
                },
                "licenses": [{
                    "id": 1,
                    "name": "Attribution-NonCommercial",
                    "url": "http://creativecommons.org/licenses/by-nc/2.0/"
                }],
                "images": [],
                "annotations": [],
                "categories": []
            }
            with open(self.output_file, 'w') as f:
                json.dump(coco_data, f, indent=4)

        
        with open(self.output_file, 'r+') as f:
            coco_data = json.load(f)
            self.category_map = {cat['name']: cat['id'] for cat in coco_data['categories']}
            image_id_map = {}
            
            for obj in self.data.findall("object"):
                class_name = obj.find('name').text
                if class_name not in self.category_map:
                    category_id = len(coco_data['categories']) + 1
                    self.category_map[class_name] = category_id
                    coco_data['categories'].append({
                        "id": category_id,
                        "name": class_name,
                        "supercategory": class_name
                    })

            
            filename = self.data.find('filename').text
            if filename not in image_id_map:
                size = self.data.find('size')
                width = int(size.find('width').text)
                height = int(size.find('height').text)

                image_id = len(coco_data['images']) + 1 
                image_id_map[filename] = image_id

                image_entry = {
                    "id": image_id,
                    "file_name": filename,
                    "width": width,
                    "height": height,
                    "date_captured": "2024-08-30"

                }
                coco_data['images'].append(image_entry)
            else:
                image_id = image_id_map[filename]
            
            for obj in self.data.findall("object"):
                class_name = obj.find('name').text
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
                area = (xmax - xmin) * (ymax - ymin)

                segmentation = []
                if 'segmentation' in obj.attrib and obj.find('segmentation').text:
                # Convert segmentation data if available
                    segmentation = [list(map(float, obj.find('segmentation').text.split()))]

                annotation_entry = {
                "id": len(coco_data['annotations']) + 1,
                "image_id": image_id,
                "category_id": self.category_map[class_name],
                "bbox": bbox,
                "area": area,
                "segmentation": segmentation,
                "iscrowd": 0,
                }               
                coco_data['annotations'].append(annotation_entry)
            
            f.seek(0)
            f.truncate()
            json.dump(coco_data, f, indent=4)


    def convert_to_yolo(self):
        yolo_annotations=[]
        size=self.data.find('size')
        width=int(size.find('width').text)
        height=int(size.find('height').text)

        category_map={}
        for obj in self.data.findall('object'):
            class_name=obj.find('name').text
            if class_name not in category_map:
                category_map[class_name]=len(category_map)

        for obj in self.data.findall('object'):
            class_name=obj.find('name').text
            class_index=category_map[class_name]
            bndbox=obj.find('bndbox')
            xmin=int(bndbox.find('xmin').text)
            ymin=int(bndbox.find('ymin').text)
            xmax=int(bndbox.find('xmax').text)
            ymax=int(bndbox.find('ymax').text)

            x_center=((xmin+xmax)/2)/width
            y_center=((ymin+ymax)/2)/height
            bbox_width=(xmax-xmin)/width
            bbox_height=(ymax-ymin)/height

            yolo_annotation=f"{class_index} {x_center:.3f} {y_center:.3f} {bbox_width:.3f} {bbox_height:.3f}"
            yolo_annotations.append(yolo_annotation)
        return yolo_annotations
    
    def save_data(self,annotations,target_format):
           
        if target_format=="YOLO":
            output_file=os.path.join(self.output_dir,os.path.splitext(os.path.basename(self.input_file))[0]+'.txt')
            with open(output_file,"w") as f:
                for annotation in annotations:
                    f.write(annotation+'\n')

class YOLOConverter(AnnotationConverter):
    def __init__(self, input_file, output_dir,image_dir=None, starting_image_id=1, starting_annotation_id=1):
        super().__init__(input_file, output_dir)
        self.image_id = starting_image_id
        self.annotation_id = starting_annotation_id
        self.image_dir=image_dir
    def load_data(self):
        with open(self.input_file,'r') as f:
            self.data=f.read().splitlines()

    def get_size(self):
        if not self.image_dir:
            raise ValueError("Image Directory not provided")
        
        image_path = os.path.normpath(self.image_dir)
        print(image_path)
        filename_without_extension = os.path.splitext(os.path.basename(self.input_file))[0]
        extensions = ['.jpg', '.jpeg', '.png']
        image_file = None
        for ext in extensions:
            potential_file = os.path.join(image_path, filename_without_extension + ext)
            if os.path.isfile(potential_file):
                image_file = potential_file
                break
            if not image_file:
                raise FileNotFoundError(f"Image file corresponding to {self.input_file} not found in directory {image_path}")
        image = cv2.imread(image_file)
        if image is None:
            raise ValueError(f"Unable to read image file {image_file}")
        
        height, width = image.shape[:2]
        print(f"Image size: width={width}, height={height}")
        return width, height,image_file
        


    def load_classes(self):
        directory = os.path.dirname(self.input_file)
        classes_path = os.path.join(directory, 'classes.txt')
        print(classes_path)
        if not os.path.exists(classes_path):
            raise FileNotFoundError("classes.txt file not found in the  directory.")
        
        with open(classes_path, 'r') as f:
            class_list = f.read().splitlines()

        return  {class_name: idx for idx,class_name in enumerate(class_list)}

    def convert_to(self,target_format,**kwargs):
        if target_format=="COCO":
            return self.convert_to_coco()
        elif target_format =="VOC":
            return self.convert_to_voc()
        else:
            raise ValueError(f"Unsupported target format: {target_format}")
   
    def convert_to_coco(self):
    
        coco_output_file = os.path.join(self.output_dir, "converted_coco_dataset.json")

        
        if os.path.exists(coco_output_file):
            with open(coco_output_file, "r") as f:
                coco_data = json.load(f)
        else:
           
            coco_data = {
                "info": {
                    "year": 2024,
                    "version": "1.0",
                    "description": "Converted COCO dataset",
                    "contributor": "Himanshi Sharma",
                    "url": "http://example.com",
                    "date_created": "2024-09-01"
                },
                "licenses": [{
                    "id": 1,
                    "name": "Attribution-NonCommercial",
                    "url": "http://creativecommons.org/licenses/by-nc/2.0/"
                }],
                "images": [],
                "annotations": [],
                "categories": []
            }

       
        existing_image_ids = {img['id'] for img in coco_data['images']}
        existing_annotation_ids = {ann['id'] for ann in coco_data['annotations']}
        
        if existing_image_ids:
            self.image_id = max(existing_image_ids) + 1
        if existing_annotation_ids:
            self.annotation_id = max(existing_annotation_ids) + 1
        
        
        width, height,img_file = self.get_size()
        image_file = os.path.splitext(os.path.basename(self.input_file))[0] + ".jpg"

        
        if not any(img['file_name'] == image_file for img in coco_data['images']):
            coco_data["images"].append({
                "id": self.image_id,
                "file_name": image_file,
                "height": height,
                "width": width,
                "date_captured": "2024-08-30"
            })

       
        for line in self.data:
            class_idx, x_center, y_center, bbox_width, bbox_height = map(float, line.split())
            x_min = (x_center - bbox_width / 2) * width
            y_min = (y_center - bbox_height / 2) * height
            bbox_width = bbox_width * width
            bbox_height = bbox_height * height

            coco_data["annotations"].append({
                "id": self.annotation_id,
                "image_id": self.image_id,
                "category_id": int(class_idx),
                "bbox": [x_min, y_min, bbox_width, bbox_height],
                "area": bbox_width * bbox_height,
                "segmentation": [],
                "iscrowd": 0
            })
            self.annotation_id += 1

        
        existing_category_names = {cat['name'] for cat in coco_data['categories']}
        for name, idx in self.load_classes().items():
            if name not in existing_category_names:
                coco_data['categories'].append({
                    "id": idx,
                    "name": name,
                    "supercategory": name
                })

        
        with open(coco_output_file, "w") as f:
            json.dump(coco_data, f, indent=4)
        
        self.image_id += 1

    def convert_to_voc(self):
        annotations = []
        width, height,img_file = self.get_size()
        classes = self.load_classes()
        for image_file in set([os.path.splitext(os.path.basename(self.input_file))[0]]):
            img_file=os.path.basename(img_file)
            annotation = ET.Element('annotation')
            ET.SubElement(annotation, "folder").text = self.output_dir
            ET.SubElement(annotation, "filename").text = img_file
            size = ET.SubElement(annotation, "size")
            ET.SubElement(size, 'width').text = str(width)  
            ET.SubElement(size, "height").text = str(height)  
            ET.SubElement(size, 'depth').text = "3"

            for line in self.data:
                class_idx, x_center, y_center, bbox_width, bbox_height = map(float, line.split())
                x_min = (x_center - bbox_width / 2) * width
                y_min = (y_center - bbox_height / 2) * height
                x_max = (x_center + bbox_width / 2) * width
                y_max = (y_center + bbox_height / 2) * height
                obj = ET.SubElement(annotation, "object")
                
                
                class_name = list(classes.keys())[list(classes.values()).index(int(class_idx))]  # Get the class name using class index
                ET.SubElement(obj, "name").text = class_name # Assuming class_idx corresponds to the class name index
                ET.SubElement(obj, "pose").text = "Unspecified"
                ET.SubElement(obj, "truncated").text = '0'
                ET.SubElement(obj, "difficult").text = '0'
                bndbox = ET.SubElement(obj, "bndbox")
                ET.SubElement(bndbox, "xmin").text = str(int(x_min))
                ET.SubElement(bndbox, "ymin").text = str(int(y_min))
                ET.SubElement(bndbox, "xmax").text = str(int(x_max))
                ET.SubElement(bndbox, "ymax").text = str(int(y_max))
            annotations.append(ET.ElementTree(annotation))
        return annotations

    def save_data(self, annotations, target_format):
       
        if target_format == "VOC":
            for annotation in annotations:
                xml_output_path = os.path.join(self.output_dir, os.path.splitext(annotation.find('filename').text)[0] + ".xml")
                annotation.write(xml_output_path)

           

def get_converter(input_file,output_dir,image_dir=None):
    if input_file.endswith('.json'):
        return COCOConverter(input_file,output_dir)
    elif input_file.endswith('.xml'):
        return PASCALVOCCOnverter(input_file,output_dir)
    elif input_file.endswith('.txt'):
        return YOLOConverter(input_file,output_dir,image_dir=image_dir)
    else:
        return ValueError("Unsupported file extension")

def main(input_path,target_format,output_dir):
    image_path = None
    if os.path.isdir(input_path):
        all_files = os.listdir(input_path)
        txt_files=[file for file in all_files if file.endswith('.txt') and file!= "classes.txt"]
        if txt_files:
            image_path = input("Enter corresponding image directory: ").strip()
    
        for file_name in all_files:
            if file_name=="classes.txt":
                continue
            file_path=os.path.join(input_path,file_name)
            converter=get_converter(file_path,output_dir,image_dir=image_path)
            converter.load_data()
            annotations=converter.convert_to(target_format)
            converter.save_data(annotations,target_format)
    else:
       
        converter=get_converter(input_path,output_dir,image_dir=image_path)
        converter.load_data()
        annotations=converter.convert_to(target_format)
        converter.save_data(annotations,target_format)
    print(f"Conversion to {target_format} completed successfully!")


if __name__=="__main__":
    
    ap=argparse.ArgumentParser()
    ap.add_argument("-i","--input_path",type=str,help="Path of the file or directory to be converted ",required=True )
    ap.add_argument("-t ",'--target_format',type=str,help="Name of the format to be converted in",required=True,choices=['COCO','VOC','YOLO'])
    ap.add_argument("-o",'--output_dir',type=str,help="Path to output directory")

    args=vars(ap.parse_args())
    
    if args["output_dir"]==None:
        output_dir="Outputs/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    else:
        output_dir=args["output_dir"]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    main(args["input_path"],args['target_format'],output_dir)


