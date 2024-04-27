# R-CNN Model Classification of Blood Cells
Authors: [Shayla Tran](https://github.com/shaytran), [Kyle Deng](https://github.com/kt1720), and [Matthew Angoh](https://github.com/mattangoh)

### Materials
1. [Canva Presentation Link](https://www.canva.com/design/DAGCEbQBBcA/PAIfLsvr-iG_aeB65WWUzA/edit?utm_content=DAGCEbQBBcA&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)
2. [Final Report](https://docs.google.com/document/d/1iAYaF60E2SaE3Sui7MzJriaIPZJeps9ty-6np5Zf-nQ/edit?usp=sharing)

## Table of Contents
1. [Introduction](#introduction)
2. [Methodology](#methodology)
3. [Results & Discussion](#results--discussion)
4. [Conclusion](#conclusion)

## Introduction
The integration of deep learning, particularly through Region-based Convolutional Neural Networks (R-CNNs) like the Faster R-CNN, has revolutionized medical imaging by enabling precise localization and classification of blood cells. This method overcomes traditional limitations by utilizing a region proposal network (RPN) that enhances the speed and accuracy of the model, crucial for effective diagnostics in hematological settings. By adapting and evaluating a pre-trained Faster R-CNN model on the BCCD dataset[^1], this study explores and demonstrates the efficacy and accuracy of our implemented R-CNN model on blood classification.

## Methodology
**Data Extraction & Preparation**

The BCCD dataset images were loaded along with their corresponding cell annotations through Python's PIL library and xml.etree.ElementTree. We created a custom BCCDDataset class, derived from torch.utils.data.Dataset, to manage data loading and transformation into tensor format suitable for PyTorch. The images were standardized to a uniform size and normalized for network compatibility, with augmentation techniques like flipping and rotating applied to improve the model's generalizability. This preprocessing pipeline was critical for the successful training and validation of our deep learning model.

***Code snippet:***
```
class BCCDDataset(torch.utils.data.Dataset):
    def __init__(self, root, image_set, transforms=None):
        self.root = os.path.join(root, 'BCCD')
        self.transforms = transforms
        self.imgs_dir = os.path.join(self.root, 'JPEGImages')
        self.annos_dir = os.path.join(self.root, 'Annotations')
        split_file = os.path.join(self.root, 'ImageSets', 'Main', f'{image_set}.txt')
        with open(split_file, 'r') as file:
            self.imgs_ids = file.read().splitlines()

    def __getitem__(self, idx):
        img_path = os.path.join(self.imgs_dir, self.imgs_ids[idx] + '.jpg')
        anno_path = os.path.join(self.annos_dir, self.imgs_ids[idx] + '.xml')
        img = Image.open(img_path).convert("RGB")
        target = self.parse_annotations(anno_path)
        if self.transforms:
            img, target = self.transforms(img, target)
        return img, target

    def parse_annotations(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        boxes = []
        labels = []
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            boxes.append([int(bbox.find('xmin').text), int(bbox.find('ymin').text),
                          int(bbox.find('xmax').text), int(bbox.find('ymax').text)])
            label = obj.find('name').text
            labels.append(label_map[label])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        return {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([idx])}
```

**Model Configuration, Training, Testing & Validation**

The Faster R-CNN, selected for its accuracy and speed in object detection, was enhanced with a ResNet50 backbone, fine-tuned to classify blood cells into three distinct classes plus background, making it ideal for clinical real-time analysis. Its integrated Region Proposal Network (RPN) enables quick region proposals, while the deep residual networks of ResNet50 ensure robust feature extraction without gradient issues. During training, we optimized key hyperparameters such as learning rate, momentum, and weight decay to balance model convergence and prevent overfitting. We employed metrics like loss, precision, recall, F1-score, and mean Average Precision (mAP) to quantitatively assess the model's ability to accurately predict cell types and localize cells. 

***Code snippet:***

Function for adapting the model
```
def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
```

## Results & Discussion
[]

### Model Comparison and Interpretation
[]

## Conclusion
[]

[^1]: Biryukov. (2021). Blood cells classification. Retrieved from https://www.kaggle.com/code/valentinbiryukov/blood-cells-classification 
