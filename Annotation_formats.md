# COCO (Common Objects in Context)
### Format
 JSON
### Description 
The COCO format is widely used for object detection, segmentation, and keypoint detection tasks. It includes information about the images, annotations for objects in the images(bounding boxes,segmentation masks,etc) and category labels. Each image and annotation has a unique ID, and the categories are defined separately.

### Structure:

    {
        "images":[
            {
                "file_name":"filename.jpg".
                "id":image_id,
                "width":image_width,
                "height":image_height
            }
        ],
        "annotations":[
            {
                "image_id":image_id,
                "bbox":[x,y,width,height],
                "category_id":category_id,
                "segmentation":[[x1,y1,x2,y2,....]],
                "area":area,
                "iscrowd":0 or 1,
                "id":annotation_id
            }
        ],
        "categories":[
            {
                "id":category_id,
                "name":"category_name",
                "supercategory":"supercategory_name"
            }
        ]
    }


### Example

    {
        "images":[
            {
                "file_name":"image1.jpg",
                "id":1,
                "width":640,
                "height":480
            }
        ],
        "annotations":[
            {
                "image_id":1,
                "bbox":[100,200,150,300],
                "category_id":1,
                "segmentation":[[110,210,130,230,150,250,170,270]],
                "area":45000,
                "iscrowd":0,
                "id":1
            }
        ],
        "categories":[
            {
                "id":1,
                "name":"person",
                "supercategory":"human"
            }
        ]
    }

### Detailed Description
- **images**: Contains metadata about the images in the dataset such as 'id','width','height' and 'file_name'.

- **annotations**: Holds the actual annotations for each image. Key Components:
    1. id: Unique identifier for the annotation
    2. image_id: The ID of the corresponding image.
    3. category_id: Refers to the 'id' in the 'categories' section.
    4. bbox: Bounding box [x,y,width,height] around the object.
    5. area: Area of the bounding box, used for validation and analysis.
    6. iscrowd: Indicates if the annotation is of a single object('0') or a crowd('1'). Used to handle cases where objects are grouped together, and precise bounding boxes or segmentation might be difficult to achieve.
    7. segmentation: Polygonal segmentation outlining the object. Used for more precise object detection and analysis.

- **Categories**: Lists the possible object categories. Each category has:
    1. id: Unique identifier for the category.
    2. name: Human-readable name of the category.
    3. supercategory: Grouping of similar categories(e.g. "animal").


# Pascal VOC

### Format
XML

### Description
Pascal VOC is a widely adopted format for object detection tasks. It organizes image data and annotations( bounding boxes) in an XML structure. Each annotation includes the image size, object categories and bounding box coordinates.

### Structure

    <annotation>
        <folder>folder_name</folder>
        <filename>filename.jpg</filename>
        <path>file_path</path>
        <size>
            <width>image_width</width>
            <height>image_height</height>
            <depth>image_depth</depth>
        </size>
        <object>
            <name>category_name</name>
            <pose>pose_info</pose>
            <truncated>0 or 1</truncated>
            <difficult>0 or 1</difficult>
            <bndbox>
                <xmin>x1</xmin>
                <ymin>y1</ymin>
                <xmax>x2</ymax>
                <ymax>y2</ymax>
            </bndbox>
        </object>
    </annotation>

### Example
    <annotation>
        <folder>images</folder>
        <filename>image1.jpg</filename>
        <path>/path/to/image1.jpg</path>
        <size>
            <width>640</width>
            <height>480</height>
            <depth>3</depth>
        </size>
        <object>
            <name>dog</name>
            <pose>Unspecified</pose>
            <truncated>0</truncated>
            <difficult>0</difficult>
            <bndbox>
                <xmin>100</xmin>
                <ymin>150</ymin>
                <xmax>300</ymax>
                <ymax>400</ymax>
            </bndbox>
        </object>
    </annotation>


### Detailed Description
**annotation** : Root element that contains the entire annotation
- folder: Name of the folder containing the image.
- filename: name of the image file
- size: Image dimensions. including width, height, and depth(number of channels, usually 3 for RGB).
- object: Repeated element for each object in the image
    1. name : Class label of the object.
    2. pose: Pose of the object( not commonly used , often "Unspecified").
    3. truncated: indicates if the object is truncated by the image border (0 for no, 1 for yes).
    4. bndbox: Bounding box around the object with coordinates: xmin,ymin,xmax,ymax.


# YOLO (You Only Look Once)

### Format
txt

### Description
The YOLO format is used for real-time object detection tasks. The annotations are saved in plain text file where each line represents one object. The format uses normalized values for bounding box coordinates, making it compact and efficient for real-time applications.

### structure

    class_id center_x center_y width height

### Example

    0 0.234375 0.520833 0.078125 0.104167

- Here, '0' is the class ID, and the other values represent the normalized coordinates and dimensions of the bounding box.

### Detailed Description
Each line represents an object in the image.
- 0: Class ID (integer corresponding to the object class,e.g., "dog").
- 0.234375 and 0.520833: Center coordinates of the bounding box (x_center,y_center), normalized to [0,1] range.
- 0.078125 and 0.104167: Width and Height of the bounding box, also normalized to [0,1] range.


# LabelMe

### Format
JSON

### Description
LabelMe is used for both object detection and segmentation tasks. It allows for annotating images with bounding boxes, polygons, and other shapes. The annotations are stored in a JSON format, which includes image_path, labeled shapes, and other metadata.

### Structure

    {
        "version":"version_number",
        "flags":{},
        "shapes":[
            {
                "label":"category_name",
                "points":[[x1,y1],[x2,y2],...],
                "group_id":group_id,
                "shape_type":"polygon" or "rectangle" or "circle",
                "flags":{}
            }
        ],
        "imagePath":"filename.jpg",
        "imageHeight":"image_height,
        "imageWidth":image_width
    }

### Example

    {
        "version":"4.5.6",
        "flags":{},
        "shapes":[
            {
                "label":"car",
                "points"=[[100,150],[200,250],[300,350],[400,450]],
                "group_id":null,
                "shape_type":"polygon",
                "flags":{}
            }
        ],
        "imagePath":"image1.jpg",
        "imageData":null,
        "imageHeight":480,
        "imageWidth":640
    }


### Detailed Description

- **version** : Version of the LabelMe annotation tool.
- **flags**: Any flags associated with the annotation (often empty).
- **shapes**: List of shapes annotated in the image.
    1. label: Class label of the object(e.g.,"dog")
    2. points: Coordinates of the polygon vertices outlining the object.
    3. group_id: Group ID for the object(if applicable).
    4. shape_type: Type of shape (e.g., "polygon","rectangle").
    5. flags: Additional flags for the shape.

- **imagePath**: Path to the image file.
- **imageData**: Encoded image data(optional,often null)
- **image Height/Width**: Dimensions of the image.

# TFRecord(Tensorflow)

### Format
TFRecord

### Description
TFRecord is a format used by Tensorflow for storing large datasets. It is especially useful for training deep learning models as it allows efficient reading and writing of large amounts of data. The annotations, along with image data, are stored in a serialized format.


### Structure

    {
        "image/filename":"filename.jpg",
        "image/encoded": encoded_image,
        "image/format": "jpg" or "png",
        "image/object/bbox/xmin": [xmin],
        "image/object/bbox/ymin": [ymin],
        "image/object/bbox/xmax": [xmax],
        "image/object/bbox/ymax": [ymax],
        "image/object/class/label": [class_id]
    }

- The encoded image data is typically in base64 format. 


### Detailed Description

- image/encoded: The raw bytes of the image file.
- image/filename: The filename of the images as a string.
- image/object/class/label: The class label as an integer.
- image/object/bbox/xmin,xmax,ymin,ymax: The normalized bounding box coordinates

# CSV (CUSTOM)

### Format
CSV

### Description
CSV is a simple, custom format used in various computer vision tasks. It stores image annotations in a tabular format where each row represents one object in an image. This format is easy to use and can be adapted to different tasks by including additional columns as needed.

### Structure

    filename,width,height,class,xmin,ymin,xmax,ymax

- **filename**: Name of the image file.
- **width**: Width of the image.
- **height**: Height of the image.
- **class**: Class label of the object (e.g., "dog").
- **xmin, ymin, xmax, ymax**: Bounding box coordinates specifying the top-left (xmin, ymin) and bottom-right (xmax, ymax) corners of the object.

# CityScapes Format

### Format
JSON

### Description
The CityScapes format is specifically designed for semantic segmentation tasks, where each pixel in an image is assigned to a class label. It is commonly used in urban scene understanding, particularly in autonomous driving research. The dataset includes annotations for different object classes like cars, pedestrians,roads,etc

This format contains high-resolution images of urban environments, along with fine-grained annotations in the form of polygonal shapes for segmentation tasks. Annotations include instance segmentation and semantic segmentation maps, as well as corresponding JSON files with metadata and polygon coordinates for each object.

### Structure
- For Instance Segmentation:

        {
            "imgHeight":image_height,
            "imgWidth":image_width,
            "objects":[
                {
                    "label": "category_name",
                    "polygon": [[x1,y1],[x2,y2],...],
                    "id": object_id
                },
                ....
            ]
        }

- For Semantic Segmentation:

        {
            "imgHeight":image_height,
            "imgWidth":image_width,
            "objects":[
                {
                    "label": "category_name",
                    "polygon": [[x1,y1],[x2,y2],...],
                    
                },
                ....
            ]
        }


### Directory structure
- **leftImg8bit/**: Contains original images.
    - train/,val/,test/: Subdirectories for training, validation and testing images.
    - image_name.png :Image files

- **gtFine/**: Contains ground truth annotations.
    - train/,val/test/: Subdirectories for corresponding images.
    - image_name_gtFine_labelIds.png: Semantic segmentation label IDs.
    - image_name_gtFine_instanceIds.png: Instance segmentation IDs.
    - image_name_gtFine_polygons.json': Polygon annotations.
    - image_name_gtFine_color.png: Color-coded segmentation images(optional)

### Directory example

    Cityscapes/
    ├── leftImg8bit/
    │   ├── train/
    │   │   ├── aachen/
    │   │   │   ├── aachen_000000_000019_leftImg8bit.png
    │   │   │   └── ...
    │   ├── val/
    │   └── test/
    ├── gtFine/
    │   ├── train/
    │   │   ├── aachen/
    │   │   │   ├── aachen_000000_000019_gtFine_labelIds.png
    │   │   │   ├── aachen_000000_000019_gtFine_instanceIds.png
    │   │   │   ├── aachen_000000_000019_gtFine_polygons.json
    │   │   │   └── ...
    │   ├── val/
    │   └── test/




### Example
- For Instance Segmentation: 

    {
        "imgHeight":1024,
        "imgWidth": 2048,
        "objects":[
            {
                "label":"car",
                "polygon": [[100, 200], [150, 250], [200, 200]],
                "id": 1
            },
            {
                "label": "person",
                "polygon": [[300, 400], [320, 420], [340, 400]],
                "id":2
            }
        ]
    }


- For Semantic Segmentation:

    {
    "imgHeight": 1024,
    "imgWidth": 2048,
    "objects": [
        {
        "label": "car",
        "polygon": [[100, 200], [150, 250], [200, 200]]
        },
        {
        "label": "car",
        "polygon": [[300, 400], [350, 450], [400, 400]]
        },
        {
        "label": "person",
        "polygon": [[500, 600], [520, 620], [540, 600]]
        }
    ]
    }


