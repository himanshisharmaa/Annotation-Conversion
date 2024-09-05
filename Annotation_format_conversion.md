# COCO
## COCO to Pascal VOC

### COCO Format:
- JSON with bbox coordinates **[x_min,y_min,width,height]**.
### Passcal VOC Format:
- XML with bounding box coordinates **[xmin,ymin,xmax,ymax]**.

### Step-by-Step Conversion:
1. Parse the COCO JSON:
    - load the JSON file and extract annotations.

2. Convert the bounding boxes:
    - Convert COCO's **[x_min,y_min,width,height]** to Pascal VOC's **[x_min,y_min,x_max,y_max]**.

            def coco_to_pascal_voc(x1,y1,w,h):
                return [x1,y1,x1+w,y1+h]

3. Create Pascal VOC XML:
    - Write the XML structure for each image.

4. Save XML files:
    - Save the XML file for each image with the corresponding annotation.


## COCO to YOLO
### COCO Format:
- JSON with bounding box coordinates [x_min, y_min, width, height].

### YOLO Format:
- TXT with bounding box coordinates **[class_id,x_center,y_center,width,height]**,normalized to the image size.

### Step-by-Step Conversion:
1. Parse the COCO JSON:
- Load the JSON file and extract annotations.

2. Convert Bounding Boxes:
- Convert COCO's **[x_min, y_min, width, height]** to YOLO format **[x_center, y_center, width, height]** normalized to the image dimensions.

    - x_center=(x_min+width/2)/img_width
        - x_min + width / 2: Calculates the x-coordinate of the center of the bounding box. x_min is the x-coordinate of the top-left corner, and adding width / 2 shifts this point to the center of the box.
        - / img_width: Normalizes the center x-coordinate by dividing it by the image's width.
    
    - y_center=(y_min+height/2)/img_height
        - y_min + height / 2: Calculates the y-coordinate of the center of the bounding box. y_min is the y-coordinate of the top-left corner, and adding height / 2 shifts this point to the center of the box.
        - / img_height: Normalizes the center y-coordinate by dividing it by the image's height.


    - width/=img_width
        - Divides the bounding box's width by the image width to normalize it, resulting in a value between 0 and 1. This step transforms the width from pixel units into a relative scale with respect to the image's width.

    - height/=img_height
        - Divides the bounding box's height by the image height to normalize it, resulting in a value between 0 and 1. This step transforms the height from pixel units into a relative scale with respect to the image's height.

3. Write YOLO Txt files:
- Save the annotations in YOLO format for each image.


# Pascal VOC
## Pascal VOC to COCO
### Pascal VOC Format:
- XML with bounding box coordinates [xmin, ymin, xmax, ymax].

### COCO Format:
- JSON with bounding box coordinates [x_min, y_min, width, height].

### Step-by-Step conversion:
1. Parse Pascal VOC XML:
- Load the XML file and extract bounding boxes and class names.

2. Convert Bounding Boxes:
- Convert Pascal VOC’s **[xmin, ymin, xmax, ymax]** to COCO’s **[x_min, y_min, width, height]**.

        def pascal_voc_to_coco(x1, y1, x2, y2):
            return [x1,y1, x2 - x1, y2 - y1]

3. Create COCO JSON Structure:
- Populate a COCO-style JSON structure with the converted annotations.

4. Save COCO JSON:
- Save the annotations in a single JSON file.

## Pascal VOC to YOLO
### Step-by-Step conversion
1. Load the Pascal VOC XML file.
2. Extract the relevant data:
    - Image details (filename, size).
    - Bounding box information.
    - Class labels.
3. Convert the Pascal VOC bounding box to YOLO format.
    - Calculate the center of the bounding box and normalize all coordinates by the image dimensions.

            x_center = (xmin + xmax)/2.0 /image_width
            y_center = (ymin + ymax)/2.0 /image_height
            box_width = (xmax - xmin) / image_width
            box_height = (ymax - ymin) / image_height

4. Save the YOLO annotations in a .txt file.

# YOLO
## YOLO to COCO
### YOLO Format
- Bounding boxes are normalized relative to the image size.
- Format: [class_id, x_center, y_center, width, height].

### COCO format

- Bounding boxes are in the format [x_min, y_min, width, height].
- Stored as a JSON file with a specific structure, including images, annotations, and categories.

### Step-by-Step conversion
1. Load the YOLO .txt annotation file.
2. Extract the relevant data:
    - Image details (filename, size).
    - Bounding box information.
    - Class IDs.
3. Denormalize YOLO coordinates and convert to COCO format **[x_min, y_min, width, height]**.

        def yolo_to_coco(x_center, y_center, w, h,  image_w, image_h):
            w = w * image_w
            h = h * image_h
            x1 = ((2 * x_center * image_w) - w)/2
            y1 = ((2 * y_center * image_h) - h)/2
            return [x1, y1, w, h]

4. Generate the COCO JSON structure.
5. Save as a JSON file.

## YOLO to Pascal VOC
### Pascal VOC Format Overview:
- Bounding boxes are in the format [xmin, ymin, xmax, ymax].
- Stored as an XML file.

### Step-by-Step conversion
1. Load the YOLO .txt annotation file.
2. Extract the relevant data:
    - Image details (filename, size).
    - Bounding box information.
    - Class IDs.
3. Denormalize YOLO coordinates and convert to Pascal VOC format **[xmin, ymin, xmax, ymax]**.

        def yolo_to_pascal_voc(x_center, y_center, w, h,  image_w, image_h):
            w = w * image_w
            h = h * image_h
            x1 = ((2 * x_center * image_w) - w)/2
            y1 = ((2 * y_center * image_h) - h)/2
            x2 = x1 + w
            y2 = y1 + h
            return [x1, y1, x2, y2]

4. Generate and save the XML file for each image.