
!pip install gradio transformers torch torchvision numpy Pillow kagglehub colorama langchain langchain-community faiss-cpu sentence-transformers huggingface-hub pandas matplotlib seaborn

# System & Utility
import os
import warnings
from io import BytesIO

# Data Handling
import numpy as np
import pandas as pd
from PIL import Image

# PyTorch for CV
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# Visualization
import matplotlib.pyplot as plt

# Console Styling
from colorama import Fore

# Colab Utilities
from google.colab import files
import kagglehub

# LangChain & RAG Components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Gradio Interface
import gradio as gr

warnings.filterwarnings('ignore')

# Download dataset (you can comment this out if already downloaded)
path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
print("Path to dataset files:", path)

Root_dir = "/root/.cache/kagglehub/datasets/vipoooool/new-plant-diseases-dataset/versions/2"
print("Contents of Root Directory:", os.listdir(Root_dir))

# Set up paths
train_dir = Root_dir + "/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"
valid_dir = Root_dir + "/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid"
test_dir = Root_dir + "/test"
Diseases_classes = os.listdir(train_dir)

print(Fore.GREEN + str(Diseases_classes))
print("\nTotal number of classes are: ", len(Diseases_classes))

# for calculating the accuracy
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# Base class for image classification model
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))

# Convolution block with BatchNormalization
def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

# ResNet architecture
class CNN_NeuralNet(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()

        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))

        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)

        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                       nn.Flatten(),
                                       nn.Linear(512, num_diseases))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

# For moving data into GPU (if available)
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# For moving data to device (CPU or GPU)
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

# For loading data in the device
class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dataloader:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dataloader)

# Load model for inference
def load_model():
    device = get_default_device()
    train = ImageFolder(train_dir, transform=transforms.ToTensor())
    model = CNN_NeuralNet(3, len(train.classes))

    # Load pretrained weights
    try:
        model.load_state_dict(torch.load('plant_disease_model.pth'))
        print(Fore.GREEN + "Model loaded successfully!")
    except:
        print(Fore.RED + "Model weights not found. Running with untrained model.")
        # You would need to train the model here or provide pretrained weights

    model = to_device(model, device)
    model.eval()
    return model, train.classes, device

# Predict class from image
def predict_image(img, model, classes, device):
    """Converts image to tensor and returns the predicted class with highest probability"""
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    # Retrieve the class label
    return classes[preds[0].item()]

# ================== RAG Chatbot Setup ==================

# Create disease information knowledge base
def create_knowledge_base():
    # Sample information about plant diseases
    disease_info = {
        "Apple___Apple_scab": {
        "description": "Apple scab is a fungal disease caused by Venturia inaequalis that affects apple trees. It is one of the most common and serious diseases of apples worldwide.",
        "symptoms": "Dark olive-green spots on leaves that later turn brown and become corky. Infected fruit have similar spots and may become deformed. Severe infections can cause premature leaf drop and stunted tree growth.",
        "treatment": "Apply fungicides in early spring when buds first appear and continue applications according to label instructions. Remove and destroy fallen leaves to reduce fungal spores. Prune infected branches during dormant season.",
        "prevention": "Plant resistant varieties such as Liberty, Enterprise, or Williams Pride. Ensure good air circulation by proper pruning. Apply preventative fungicide sprays before rainy periods. Maintain orchard sanitation by removing fallen leaves and fruit."
    },
    "Apple___Black_rot": {
        "description": "Black rot is a fungal disease caused by Botryosphaeria obtusa that affects apple trees, causing leaf spots, fruit rot, and cankers on branches.",
        "symptoms": "Purple spots on leaves that gradually expand. Infected fruit develop concentric rings of rot with black fruiting bodies. Branches may develop cankers that can girdle and kill limbs.",
        "treatment": "Remove infected plant material including mummified fruit and cankers. Apply fungicides during the growing season, especially during periods of warm, wet weather. Prune out diseased branches, cutting at least 8 inches below visible infection.",
        "prevention": "Prune trees to improve air circulation. Clean up fallen fruit and leaves. Maintain tree vigor through proper fertilization and irrigation. Control insects that may create entry wounds. Use resistant varieties when possible."
    },
    "Apple___Cedar_apple_rust": {
        "description": "Cedar apple rust is a fungal disease caused by Gymnosporangium juniperi-virginianae that requires both apple trees and cedar trees to complete its life cycle.",
        "symptoms": "Bright orange-yellow spots on leaves and fruit. Leaf spots often have a red border. Severely infected leaves may drop prematurely. Small, raised spots may develop on the fruit surface.",
        "treatment": "Apply fungicides in spring as flower buds begin to show color. Remove galls from nearby cedar trees. Avoid planting apple trees near cedar trees.",
        "prevention": "Plant resistant apple varieties like Liberty, Jonafree, or Enterprise. If possible, remove cedar trees within a 2-mile radius. Apply protective fungicides starting at pink bud stage through petal fall."
    },
    "Apple___healthy": {
        "description": "A healthy apple tree shows proper growth and development without signs of disease or pest damage.",
        "symptoms": "No symptoms of disease. Leaves are vibrant green and properly developed. Fruit develops normally without spots or deformities.",
        "treatment": "No treatment needed. Continue regular care.",
        "prevention": "Regular pruning to maintain tree shape and air circulation. Balanced fertilization based on soil tests. Proper watering, especially during dry periods. Routine monitoring for early signs of pests or diseases."
    },
    "Blueberry___healthy": {
        "description": "A healthy blueberry plant shows proper growth and development without signs of disease or pest damage.",
        "symptoms": "No disease symptoms. Leaves are green and properly developed. Berries are uniform in size and color. New growth is vigorous.",
        "treatment": "No treatment needed. Continue regular care.",
        "prevention": "Maintain proper soil pH between 4.5-5.5. Provide adequate mulch to retain moisture. Water consistently but avoid waterlogging. Prune annually to remove old canes and promote new growth."
    },
    "Cherry_(including_sour)___healthy": {
        "description": "A healthy cherry tree shows proper growth and development without signs of disease or pest damage.",
        "symptoms": "No disease symptoms. Leaves are green, glossy, and properly developed. Fruit develops normally without spots or deformities.",
        "treatment": "No treatment needed. Continue regular care.",
        "prevention": "Regular pruning to maintain tree shape and air circulation. Balanced fertilization based on soil tests. Proper watering, especially during dry periods. Routine monitoring for early signs of pests or diseases."
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "description": "Powdery mildew is a fungal disease caused by Podosphaera clandestina that affects cherry trees, particularly in humid conditions.",
        "symptoms": "White powdery coating on leaves, shoots, and sometimes fruit. Affected leaves may curl upward, become distorted, and drop prematurely. New growth may be stunted.",
        "treatment": "Apply fungicides at the first sign of infection. Sulfur-based products are effective for mild infections. Systemic fungicides may be needed for severe cases. Prune out heavily infected areas.",
        "prevention": "Plant resistant varieties when possible. Improve air circulation by proper pruning and appropriate spacing. Avoid excessive nitrogen fertilization which promotes susceptible new growth. Water at the base of plants rather than overhead."
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "description": "Gray leaf spot is a fungal disease caused by Cercospora zeae-maydis that affects corn plants, particularly in warm, humid conditions.",
        "symptoms": "Rectangular, gray to tan lesions that run parallel to leaf veins. Lesions may coalesce in severe infections, causing entire leaves to blight. Lower leaves are usually affected first.",
        "treatment": "Apply fungicides if detected early and if economically justified. Consider applications when disease appears before tasseling in seed corn.",
        "prevention": "Plant resistant hybrids. Practice crop rotation with non-host crops. Tillage can reduce inoculum from crop residue. Maintain balanced soil fertility. Avoid late planting which increases disease risk."
    },
    "Corn_(maize)___Common_rust_": {
        "description": "Common rust is caused by the fungus Puccinia sorghi and affects corn crops worldwide, developing in cool, moist conditions.",
        "symptoms": "Small, circular to elongated cinnamon-brown pustules on both leaf surfaces. Pustules may turn dark brown to black as they mature. Severe infections can cause leaf yellowing and premature death.",
        "treatment": "Apply fungicides if detected early in the growing season and if economically justified. Applications are most effective when made at the first sign of disease.",
        "prevention": "Plant resistant hybrids. Early planting can help avoid severe infections. Maintain proper soil fertility based on soil tests. Monitor fields regularly for early detection."
    },
    "Corn_(maize)___healthy": {
        "description": "A healthy corn plant shows proper growth and development without signs of disease or pest damage.",
        "symptoms": "No disease symptoms. Leaves are green and properly developed. Stalks are strong and upright. Ears develop normally.",
        "treatment": "No treatment needed. Continue regular care.",
        "prevention": "Proper crop rotation. Adequate plant spacing. Balanced fertility based on soil tests. Timely planting. Weed control to reduce competition. Regular monitoring for pests and diseases."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "description": "Northern leaf blight is a fungal disease of corn caused by Exserohilum turcicum that can significantly reduce yield in susceptible hybrids.",
        "symptoms": "Long, elliptical lesions that are grayish-green to tan in color. Lesions may be 1-6 inches long and usually begin on lower leaves. Dark spores may be visible on lesions during humid conditions.",
        "treatment": "Fungicide application during early stages of the disease, particularly before tasseling. Timing is critical for effective control.",
        "prevention": "Plant resistant hybrids. Practice crop rotation with non-host crops. Residue management through tillage when practical. Maintain balanced soil fertility. Avoid overhead irrigation which can spread spores."
    },
    "Grape___Black_rot": {
        "description": "Black rot is a fungal disease caused by Guignardia bidwellii that affects grapes, causing significant crop losses in warm, humid regions.",
        "symptoms": "Brown circular lesions on leaves with a darker border. Tan spots with black dots (pycnidia) on fruit that eventually shrivel into black, mummified berries. Lesions may also appear on shoots and petioles.",
        "treatment": "Apply fungicides beginning at bud break and continuing on a 10-14 day schedule until veraison. Remove mummified berries, infected leaves, and canes during dormant season.",
        "prevention": "Prune vines to improve air circulation. Remove all mummified berries. Manage weeds to reduce humidity around vines. Apply protective fungicides before rainy periods. Use resistant varieties when possible."
    },
    "Grape___Esca_(Black_Measles)": {
        "description": "Esca, also known as Black Measles, is a complex fungal disease affecting grapevines, caused by several fungi including Phaeomoniella chlamydospora and Phaeoacremonium species.",
        "symptoms": "Interveinal chlorosis and necrosis on leaves, creating a tiger-stripe pattern. Red varieties may show red discoloration. Berries develop irregular, dark spotting and may shrivel. Wood shows dark streaking when cut.",
        "treatment": "No effective fungicide treatments exist. Remove and destroy infected vines. Protect pruning wounds with fungicide paste or paint. Avoid pruning during wet weather.",
        "prevention": "Use disease-free planting material. Prune during dry weather. Seal large pruning wounds. Avoid water stress. Practice balanced fertilization. Remove infected vines promptly."
    },
    "Grape___healthy": {
        "description": "A healthy grapevine shows proper growth and development without signs of disease or pest damage.",
        "symptoms": "No disease symptoms. Leaves are green and properly developed. Fruit clusters develop normally without spots or shriveling.",
        "treatment": "No treatment needed. Continue regular care.",
        "prevention": "Regular pruning to maintain vine shape and air circulation. Balanced fertilization based on soil tests. Proper watering, especially during dry periods. Routine monitoring for early signs of pests or diseases."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "description": "Leaf blight, also known as Isariopsis Leaf Spot, is caused by the fungus Pseudocercospora vitis and affects grapevines, particularly in warm, humid conditions.",
        "symptoms": "Small, dark brown to black spots that may coalesce to form larger necrotic areas. Lesions may have light centers with dark margins. Severe infections can cause premature defoliation.",
        "treatment": "Apply fungicides at the first sign of disease. Remove and destroy infected leaves. Ensure good air circulation through proper pruning.",
        "prevention": "Prune vines to improve air circulation. Practice proper canopy management. Apply preventative fungicides during periods of high humidity. Avoid overhead irrigation. Control weeds to reduce humidity around vines."
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "description": "Huanglongbing (HLB), or citrus greening, is a bacterial disease caused by Candidatus Liberibacter asiaticus, spread by the Asian citrus psyllid. It is one of the most serious citrus diseases worldwide.",
        "symptoms": "Blotchy mottling of leaves that is asymmetrical across the leaf midrib. Yellow shoots in an otherwise green canopy. Misshapen, bitter fruit that remains green at the bottom. Fruit drop and twig dieback.",
        "treatment": "No effective cure exists. Remove and destroy infected trees to prevent spread. Control psyllid populations with insecticides. Nutritional treatments may help manage symptoms but do not cure the disease.",
        "prevention": "Plant certified disease-free trees. Monitor for psyllids and control them promptly. Inspect trees regularly for symptoms. Avoid moving plant material from infected areas. Support research for resistant varieties."
    },
    "Peach___Bacterial_spot": {
        "description": "Bacterial spot is caused by Xanthomonas arboricola pv. pruni and affects peach trees, particularly in warm, humid regions.",
        "symptoms": "Small water-soaked spots on leaves that develop into angular, purple-brown lesions. Spots may fall out, creating a 'shot-hole' appearance. Fruit develops water-soaked spots that become slightly raised and cracked.",
        "treatment": "Apply copper-based bactericides starting at leaf-out. Prune out infected branches during dormant season. Maintain good tree health through proper fertilization and irrigation.",
        "prevention": "Plant resistant varieties like Elberta or Redhaven. Avoid overhead irrigation. Ensure good air circulation through proper pruning. Practice crop rotation in nurseries. Apply preventative copper sprays before rainy periods."
    },
    "Peach___healthy": {
        "description": "A healthy peach tree shows proper growth and development without signs of disease or pest damage.",
        "symptoms": "No disease symptoms. Leaves are green and properly developed. Fruit develops normally without spots or deformities.",
        "treatment": "No treatment needed. Continue regular care.",
        "prevention": "Regular pruning to maintain tree shape and air circulation. Balanced fertilization based on soil tests. Proper watering, especially during dry periods. Routine monitoring for early signs of pests or diseases."
    },
    "Pepper,_bell___Bacterial_spot": {
        "description": "Bacterial spot is caused by Xanthomonas species and affects bell peppers, particularly in warm, humid conditions.",
        "symptoms": "Small, water-soaked lesions on leaves that enlarge and turn brown with yellow halos. Spots may merge to form irregular blotches. Fruit develops raised, scabby spots that can crack and allow secondary infections.",
        "treatment": "Apply copper-based bactericides early in the season. Remove and destroy infected plant material. Avoid working with plants when wet.",
        "prevention": "Use disease-free seeds and transplants. Practice crop rotation (3-4 years). Avoid overhead irrigation. Space plants adequately for good air circulation. Avoid handling plants when wet."
    },
    "Pepper,_bell___healthy": {
        "description": "A healthy bell pepper plant shows proper growth and development without signs of disease or pest damage.",
        "symptoms": "No disease symptoms. Leaves are green and properly developed. Fruit develops normally without spots or deformities.",
        "treatment": "No treatment needed. Continue regular care.",
        "prevention": "Proper spacing for good air circulation. Balanced fertilization based on soil tests. Consistent watering, avoiding drought stress. Mulching to prevent soil splash. Routine monitoring for early signs of pests or diseases."
    },
    "Potato___Early_blight": {
        "description": "Early blight is a fungal disease caused by Alternaria solani that affects potato plants, particularly in warm, humid conditions.",
        "symptoms": "Dark brown to black spots with concentric rings (target spots) on older leaves first. Lesions may coalesce, causing leaf yellowing and premature defoliation. Stems can develop dark, sunken lesions.",
        "treatment": "Apply fungicides at the first sign of disease. Remove and destroy infected plant material. Maintain good plant health through proper fertilization and irrigation.",
        "prevention": "Practice crop rotation (3-4 years). Use certified disease-free seed potatoes. Space plants adequately for good air circulation. Hill soil around plants to reduce spore splash. Apply preventative fungicides during periods of high humidity."
    },
    "Potato___healthy": {
        "description": "A healthy potato plant shows proper growth and development without signs of disease or pest damage.",
        "symptoms": "No disease symptoms. Leaves are green and properly developed. Stems are strong and upright. Tubers develop normally without blemishes.",
        "treatment": "No treatment needed. Continue regular care.",
        "prevention": "Use certified disease-free seed potatoes. Practice crop rotation. Proper hilling to prevent greening. Consistent watering, avoiding drought stress. Harvest when mature to prevent tuber damage."
    },
    "Potato___Late_blight": {
        "description": "Late blight is a devastating fungal disease caused by Phytophthora infestans that can rapidly destroy potato crops under favorable conditions.",
        "symptoms": "Irregular dark brown to black water-soaked spots on leaves and stems, often with a pale green border. White fuzzy mold appears on the underside of leaves in humid conditions. Infected tubers develop reddish-brown to purplish discoloration and rot.",
        "treatment": "Remove and destroy all infected plant material immediately. Apply appropriate fungicides containing chlorothalonil, mancozeb, or copper-based products. Continue preventative fungicide applications on a 7-10 day schedule during high-risk periods.",
        "prevention": "Plant resistant varieties. Use certified disease-free seed potatoes. Provide good air circulation by proper spacing. Avoid overhead irrigation. Practice crop rotation (3-4 years). Destroy volunteer potatoes. Hill soil around plants. Remove all plant debris after harvest."
    },
    "Raspberry___healthy": {
        "description": "A healthy raspberry plant exhibits vigorous growth with proper cane development and fruit production.",
        "symptoms": "No disease symptoms. Leaves are green and properly formed. Canes are strong with normal color. Flowers and fruit develop normally. Root system is well-established.",
        "treatment": "No treatment needed. Continue regular care.",
        "prevention": "Use certified disease-free planting stock. Provide proper spacing for air circulation. Prune to maintain plant vigor. Water at the base of plants. Apply balanced fertilizer according to soil tests. Remove old fruiting canes after harvest."
    },
    "Soybean___healthy": {
        "description": "A healthy soybean plant shows proper growth characteristics with uniform leaf development and normal pod formation.",
        "symptoms": "No disease symptoms. Leaves are green and properly formed. Stems are strong and upright. Pods develop normally with healthy seeds. Root system includes nitrogen-fixing nodules.",
        "treatment": "No treatment needed. Continue regular care.",
        "prevention": "Use certified disease-free seeds. Practice crop rotation with non-legume crops. Maintain proper plant spacing. Ensure adequate drainage. Conduct regular soil testing. Inoculate seeds with rhizobia bacteria if planting in new fields."
    },
    "Squash___Powdery_mildew": {
        "description": "Powdery mildew is a common fungal disease affecting squash plants, caused by several fungi including Podosphaera xanthii and Erysiphe cichoracearum.",
        "symptoms": "White powdery fungal growth on leaf surfaces, stems, and sometimes fruit. Affected leaves may yellow, curl, and eventually die. Severe infections can reduce photosynthesis and yield.",
        "treatment": "Apply fungicides containing sulfur, potassium bicarbonate, or neem oil at first signs of infection. Remove heavily infected leaves. Apply milk spray (1:10 ratio of milk to water) as an organic alternative.",
        "prevention": "Plant resistant varieties. Provide adequate spacing for good air circulation. Avoid overhead watering. Water in the morning so leaves dry quickly. Apply preventative fungicides during humid weather. Practice crop rotation. Remove plant debris after harvest."
    },
    "Strawberry___healthy": {
        "description": "A healthy strawberry plant displays proper growth habits with lush foliage and normal fruit development.",
        "symptoms": "No disease symptoms. Leaves are green and properly formed. Runners develop normally. Flowers and fruit form properly. Root system is well-established.",
        "treatment": "No treatment needed. Continue regular care.",
        "prevention": "Use certified disease-free plants. Renew plantings every 3-4 years. Provide proper spacing. Mulch around plants to reduce soil splash. Water at the base of plants. Remove runners as needed. Apply balanced fertilizer based on soil tests."
    },
    "Strawberry___Leaf_scorch": {
        "description": "Leaf scorch is a fungal disease caused by Diplocarpon earlianum that affects strawberry plants, particularly in warm, wet conditions.",
        "symptoms": "Small, dark purple to reddish-purple spots appear on upper leaf surfaces. Spots expand and develop tan centers with purple borders. Leaves develop dark brown edges and may dry out completely, appearing scorched. Severe infections can reduce yield and plant vigor.",
        "treatment": "Remove and destroy infected leaves. Apply fungicides containing captan, myclobutanil, or copper-based products. Ensure proper fertilization to maintain plant vigor.",
        "prevention": "Plant resistant varieties. Avoid overhead irrigation. Provide adequate spacing for good air circulation. Apply mulch to prevent soil splash. Renovate beds annually. Practice crop rotation. Remove old leaves after harvest. Apply preventative fungicides during wet periods."
    },
    "Tomato___Bacterial_spot": {
        "description": "Bacterial spot is caused by Xanthomonas species bacteria and affects tomato foliage, stems, and fruit, especially in warm, wet conditions.",
        "symptoms": "Small, dark brown to black circular spots with yellow halos on leaves. Spots may merge, causing leaf blight. Fruit develops small, raised, dark brown spots that may have yellow halos. Severe infections can cause significant defoliation and reduced yield.",
        "treatment": "Remove and destroy infected plant parts. Apply copper-based bactericides or streptomycin according to label instructions. Avoid working with plants when wet.",
        "prevention": "Use disease-free seeds and transplants. Practice crop rotation (2-3 years). Avoid overhead irrigation. Provide adequate spacing for air circulation. Remove plant debris after harvest. Apply preventative bactericides during high-risk periods. Use resistant varieties when available."
    },
    "Tomato___Early_blight": {
        "description": "Early blight is a fungal disease caused by Alternaria solani that affects tomato plants, particularly older leaves first.",
        "symptoms": "Dark brown to black spots with concentric rings (target-like pattern) on older leaves. Yellow areas surrounding the spots. Affected leaves may turn yellow and drop. Stem lesions may occur, appearing dark and sunken. Fruit can develop dark, leathery spots near the stem.",
        "treatment": "Remove and destroy infected leaves. Apply fungicides containing chlorothalonil, mancozeb, or copper-based products on a 7-10 day schedule. Ensure proper nutrition to maintain plant vigor.",
        "prevention": "Practice crop rotation (3-4 years). Use disease-free seeds and transplants. Provide adequate spacing for air circulation. Mulch around plants to prevent soil splash. Stake or cage plants to keep foliage off the ground. Water at the base of plants. Apply balanced fertilizer based on soil tests."
    },
    "Tomato___healthy": {
        "description": "A healthy tomato plant exhibits proper growth with vibrant green foliage and normal fruit development.",
        "symptoms": "No disease symptoms. Leaves are uniformly green and properly formed. Stems are strong and upright. Flowers and fruit develop normally. Root system is well-established.",
        "treatment": "No treatment needed. Continue regular care.",
        "prevention": "Use disease-free seeds and transplants. Practice crop rotation. Provide proper spacing for air circulation. Stake or cage plants. Water at the base of plants. Apply balanced fertilizer based on soil tests. Mulch around plants. Remove lower leaves that touch the soil."
    },
    "Tomato___Late_blight": {
        "description": "Late blight is a devastating fungal disease caused by Phytophthora infestans that can rapidly destroy tomato plants under favorable conditions.",
        "symptoms": "Irregular, water-soaked, greasy-looking dark brown to black spots on leaves and stems. White fuzzy mold may appear on the underside of leaves in humid conditions. Rapid browning and wilting of foliage. Fruit develops large, firm, dark brown greasy spots.",
        "treatment": "Remove and destroy all infected plant material immediately. Apply fungicides containing chlorothalonil, mancozeb, or copper-based products. Continue preventative fungicide applications on a 5-7 day schedule during high-risk periods.",
        "prevention": "Plant resistant varieties. Use disease-free seeds and transplants. Provide good air circulation by proper spacing. Avoid overhead irrigation. Practice crop rotation (3-4 years). Remove volunteer tomatoes and related plants. Apply preventative fungicides during cool, wet weather."
    },
    "Tomato___Leaf_Mold": {
        "description": "Leaf mold is a fungal disease caused by Passalora fulva (formerly Fulvia fulva or Cladosporium fulvum) that thrives in high humidity conditions.",
        "symptoms": "Pale green to yellow spots on upper leaf surfaces that turn into brownish-yellow spots. Olive-green to grayish-brown velvety mold growth on the undersides of leaves. Affected leaves may curl, wither, and drop. Severe infections can significantly reduce yield.",
        "treatment": "Remove and destroy infected leaves. Improve air circulation by pruning dense foliage. Apply fungicides containing chlorothalonil, mancozeb, or copper-based products according to label instructions.",
        "prevention": "Plant resistant varieties. Reduce humidity in greenhouses or high tunnels. Provide adequate spacing for air circulation. Avoid overhead irrigation. Prune to improve air flow. Remove lower leaves that are close to the soil. Apply preventative fungicides during high-risk periods."
    },
    "Tomato___Septoria_leaf_spot": {
        "description": "Septoria leaf spot is a fungal disease caused by Septoria lycopersici that primarily affects tomato foliage.",
        "symptoms": "Small, circular spots with gray centers and dark borders on lower leaves first. Spots may have yellow halos. Tiny black fruiting bodies (pycnidia) may be visible within the spots. Progressive yellowing and dropping of affected leaves. Severe infections can cause significant defoliation.",
        "treatment": "Remove and destroy infected leaves. Apply fungicides containing chlorothalonil, mancozeb, or copper-based products on a 7-10 day schedule. Continue applications until harvest if disease pressure is high.",
        "prevention": "Practice crop rotation (3-4 years). Use disease-free seeds and transplants. Provide adequate spacing for air circulation. Mulch around plants to prevent soil splash. Stake or cage plants to keep foliage off the ground. Water at the base of plants. Remove plant debris after harvest."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "description": "Two-spotted spider mites (Tetranychus urticae) are tiny arachnids that feed on plant cell contents, causing damage to tomato plants, especially during hot, dry conditions.",
        "symptoms": "Tiny yellow or white speckles on upper leaf surfaces. Bronzing or yellowing of leaves. Fine webbing on the undersides of leaves and between stems in severe infestations. Premature leaf drop. Reduced plant vigor and yield. Mites may be visible as tiny moving dots, especially on the undersides of leaves.",
        "treatment": "Spray plants with strong jets of water to dislodge mites. Apply insecticidal soap or horticultural oil to affected areas. For severe infestations, use miticides such as abamectin, bifenazate, or spiromesifen according to label instructions. Introduce predatory mites as biological control.",
        "prevention": "Maintain proper irrigation to avoid drought stress. Increase humidity around plants. Monitor plants regularly for early detection. Avoid excessive nitrogen fertilization. Remove heavily infested plants. Practice crop rotation. Maintain proper spacing for air circulation. Avoid using broad-spectrum insecticides that kill beneficial predators."
    },
    "Tomato___Target_Spot": {
        "description": "Target spot is a fungal disease caused by Corynespora cassiicola that affects tomato foliage, stems, and fruit.",
        "symptoms": "Circular to irregular brown spots with distinctive concentric rings (target-like pattern) on leaves. Spots may coalesce, causing leaf blight. Stems can develop elongated lesions. Fruit develops sunken, dark brown to black spots with concentric rings. Severely affected leaves may yellow and drop.",
        "treatment": "Remove and destroy infected plant parts. Apply fungicides containing chlorothalonil, mancozeb, or azoxystrobin according to label instructions. Rotate between different fungicide classes to prevent resistance.",
        "prevention": "Practice crop rotation (2-3 years). Use disease-free seeds and transplants. Provide adequate spacing for air circulation. Mulch around plants to prevent soil splash. Stake or cage plants to keep foliage off the ground. Water at the base of plants. Remove plant debris after harvest."
    },
    "Tomato___Tomato_mosaic_virus": {
        "description": "Tomato mosaic virus (ToMV) is a highly contagious viral disease that can persist in soil and plant debris for years.",
        "symptoms": "Mottled light and dark green patches on leaves. Leaf distortion, wrinkling, or narrowing. Stunted plant growth. Yellow-green mottling on fruit. Reduced fruit set and yield. Necrotic spots may develop on leaves, stems, and fruit.",
        "treatment": "No cure exists for viral infections. Remove and destroy infected plants completely. Disinfect tools, stakes, and hands after handling infected plants with a 10% bleach solution or 70% alcohol.",
        "prevention": "Use certified disease-free seeds and transplants. Select resistant varieties when available. Practice strict sanitation measures. Wash hands and disinfect tools before handling plants. Control insect vectors. Avoid handling tobacco products before working with tomatoes (tobacco mosaic virus can cross-infect). Remove weeds that may harbor the virus. Practice crop rotation."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "description": "Tomato yellow leaf curl virus (TYLCV) is a devastating viral disease transmitted primarily by whiteflies (Bemisia tabaci).",
        "symptoms": "Upward curling and yellowing of leaf edges. Leaves become small and cup upward. Severe stunting of plant growth. Flowers may drop before fruit set. Significantly reduced fruit production. Plants infected when young may produce no fruit.",
        "treatment": "No cure exists for viral infections. Remove and destroy infected plants completely. Control whitefly populations with insecticides or insecticidal soaps. Use reflective mulch to repel whiteflies.",
        "prevention": "Use certified disease-free transplants. Plant resistant varieties when available. Use yellow sticky traps to monitor whitefly populations. Apply insecticides preventatively in high-risk areas. Use reflective mulch to repel whiteflies. Install fine mesh screens in greenhouse production. Remove and destroy crop residues after harvest. Maintain a weed-free buffer zone around fields."
    }
    }

    # Convert to documents for RAG - but now we'll create one document per disease
    documents = []
    for disease, info in disease_info.items():
        doc_text = f"Disease: {disease}\n\n"
        doc_text += f"Description: {info['description']}\n\n"
        doc_text += f"Symptoms: {info['symptoms']}\n\n"
        doc_text += f"Treatment: {info['treatment']}\n\n"
        doc_text += f"Prevention: {info['prevention']}\n\n"
        documents.append(doc_text)

    # Create text splitter and split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.create_documents(documents)

    # Create embeddings and vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)

    return vectorstore, disease_info

# Set up RAG prompt with cleaner template
def setup_rag_prompt():
    template = """
    You are a helpful plant disease expert. Use the following context to answer the user's question about the detected plant disease.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}

    User Question: {question}

    Answer:
    """
    return PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

# Modified setup_rag_chatbot function
def setup_rag_chatbot(huggingface_token):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_token

    vectorstore, disease_info = create_knowledge_base()

    # Use Mistral-7B-Instruct from HuggingFace Hub
    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.1",
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )

    # Create custom prompt
    qa_prompt = setup_rag_prompt()

    # Create the conversational chain
    #qa_chain = ConversationalRetrievalChain.from_llm(
    #    llm=llm,
    #    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    #    memory=memory,
    #    combine_docs_chain_kwargs={"prompt": qa_prompt}
    #)

    return vectorstore, disease_info, llm, qa_prompt

# ================== Gradio Interface ==================

# Create integrated Gradio interface
def create_integrated_interface():
    # Initialize chat history and QA chain
    qa_chain = None

    # Process image and handle chat
    def process_image_and_chat(image, chat_history, query, huggingface_token):
        # Use global variables properly
        global model, classes, device, vectorstore, disease_info, llm, qa_prompt

        if image is None:
            return chat_history + [(query, "Please upload an image first so I can identify the plant disease.")]

        # Load model if not already loaded
        try:
            if 'model' not in globals() or model is None:
                model, classes, device = load_model()
        except Exception as e:
            return chat_history + [(query, f"Error loading model: {str(e)}. Please check if the model file exists.")]

        # Process image
        try:
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])
            img = transform(image.convert('RGB'))

            # Predict disease
            disease_name = predict_image(img, model, classes, device)
        except Exception as e:
            return chat_history + [(query, f"Error processing image: {str(e)}. Please try another image.")]

        # Initialize RAG components if not already set up
        if (not huggingface_token or huggingface_token.strip() == ""):
            return chat_history + [(query, "Please enter a valid HuggingFace token to enable the chatbot functionality.")]

        try:
            if 'vectorstore' not in globals() or vectorstore is None:
                os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_token
                vectorstore, disease_info, llm, qa_prompt = setup_rag_chatbot(huggingface_token)
        except Exception as e:
            return chat_history + [(query, f"Error initializing chatbot: {str(e)}. Please check your HuggingFace token.")]

        if query == "":
            # Initial message after image upload
            response = f"I've detected {disease_name} in your plant image. Would you like information about this disease or suggestions for treatment and prevention?"
        else:
            try:
                # Get specific disease information
                if disease_name in disease_info:
                    disease_data = disease_info[disease_name]

                    # Create a context string with only the relevant disease information
                    context = f"Disease: {disease_name}\n\n"
                    context += f"Description: {disease_data['description']}\n\n"
                    context += f"Symptoms: {disease_data['symptoms']}\n\n"
                    context += f"Treatment: {disease_data['treatment']}\n\n"
                    context += f"Prevention: {disease_data['prevention']}\n\n"

                    # Format the query for the LLM
                    formatted_query = f"For {disease_name}: {query}"

                    # Use LLM to generate a response based on the specific disease context
                    from langchain.chains import LLMChain
                    chain = LLMChain(llm=llm, prompt=qa_prompt)
                    result = chain.run(context=context, question=formatted_query)

                    response = result
                else:
                    # Fallback if disease not found in our knowledge base
                    response = f"I've detected {disease_name}, but I don't have specific information about this disease in my database."
            except Exception as e:
                # Detailed error handling
                print(f"Error in RAG processing: {str(e)}")
                response = f"I encountered an error while processing your question about {disease_name}. Error details: {str(e)}"

        # Update chat history
        return chat_history + [(query, response)]

    # Create Gradio interface
    with gr.Blocks(title="Plant Disease Detection & Advisor") as demo:
        gr.Markdown("# 🌱 Plant Disease Detection & Treatment Advisor")
        gr.Markdown("Upload a plant leaf image to identify diseases and get treatment recommendations")

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Upload Plant Image")
                image_output = gr.Image(label="Processed Image")
                token_input = gr.Textbox(
                    label="HuggingFace Token (required for chatbot)",
                    placeholder="Enter your HuggingFace token here",
                    type="password"
                )

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Conversation")
                msg = gr.Textbox(
                    placeholder="Ask about the detected disease or treatment options...",
                    label="Your Question"
                )
                with gr.Row():
                    clear_button = gr.Button("Clear")
                    submit_button = gr.Button("Submit")

        # Handle interactions
        def submit_message(image, chat_history, query, token):
            if not query.strip():
                return chat_history
            return process_image_and_chat(image, chat_history, query, token)

        submit_button.click(
            fn=submit_message,
            inputs=[image_input, chatbot, msg, token_input],
            outputs=[chatbot]
        ).then(
            fn=lambda: "",
            inputs=None,
            outputs=[msg]
        )

        msg.submit(
            fn=submit_message,
            inputs=[image_input, chatbot, msg, token_input],
            outputs=[chatbot]
        ).then(
            fn=lambda: "",
            inputs=None,
            outputs=[msg]
        )

        def clear_all():
            return [], None, ""

        clear_button.click(fn=clear_all, inputs=[], outputs=[chatbot, image_output, msg])

        # When image is uploaded, initiate conversation
        def process_image_upload(img, chat_history, token):
            if img is None:
                return None, chat_history
            new_chat_history = process_image_and_chat(img, chat_history, "", token)
            return img, new_chat_history

        image_input.change(
            fn=process_image_upload,
            inputs=[image_input, chatbot, token_input],
            outputs=[image_output, chatbot]
        )

    return demo

# Main function to run the app
def main():
    # Launch Gradio interface
    # Replace this with environment variables or secure configuration
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""  # put ur huggingface api token within the ""
    demo = create_integrated_interface()
    demo.launch(share=True)  # share=True creates a public link

if __name__ == "__main__":
    main()

# ================== Save and load model ==================

# For training the model (If needed)
def train_model():
    # Load data
    train = ImageFolder(train_dir, transform=transforms.ToTensor())
    valid = ImageFolder(valid_dir, transform=transforms.ToTensor())

    # Set device
    device = get_default_device()

    # Create data loaders
    batch_size = 32
    train_dataloader = DataLoader(train, batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valid_dataloader = DataLoader(valid, batch_size, num_workers=2, pin_memory=True)

    # Wrap in device data loaders
    train_dataloader = DeviceDataLoader(train_dataloader, device)
    valid_dataloader = DeviceDataLoader(valid_dataloader, device)

    # Define model
    model = CNN_NeuralNet(3, len(train.classes))
    model = to_device(model, device)

    # Training function
    @torch.no_grad()
    def evaluate(model, val_loader):
        model.eval()
        outputs = [model.validation_step(batch) for batch in val_loader]
        return model.validation_epoch_end(outputs)

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def fit_OneCycle(epochs, max_lr, model, train_loader, val_loader, weight_decay=0,
                grad_clip=None, opt_func=torch.optim.SGD):
        torch.cuda.empty_cache()
        history = []  # For collecting the results

        optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr,
                                                epochs=epochs, steps_per_epoch=len(train_loader))

        for epoch in range(epochs):
            # Training
            model.train()
            train_losses = []
            lrs = []
            for batch in train_loader:
                loss = model.training_step(batch)
                train_losses.append(loss)
                loss.backward()

                # gradient clipping
                if grad_clip:
                    nn.utils.clip_grad_value_(model.parameters(), grad_clip)

                optimizer.step()
                optimizer.zero_grad()

                # recording and updating learning rates
                lrs.append(get_lr(optimizer))
                sched.step()

            # validation
            result = evaluate(model, val_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            result['lrs'] = lrs
            model.epoch_end(epoch, result)
            history.append(result)

        return history

    # Train the model
    history = [evaluate(model, valid_dataloader)]

    num_epoch = 5
    lr_rate = 0.01
    grad_clip = 0.15
    weight_decay = 1e-4
    optims = torch.optim.Adam

    history += fit_OneCycle(num_epoch, lr_rate, model, train_dataloader, valid_dataloader,
                          grad_clip=grad_clip,
                          weight_decay=weight_decay,
                          opt_func=optims)

    # Save the model
    torch.save(model.state_dict(), 'plant_disease_model.pth')
    print(Fore.GREEN + "Model saved successfully!")

    return model, history

# ================== Main Function ==================

def main():
    # Check if model exists, if not train it
    model_path = 'plant_disease_model.pth'
    if not os.path.exists(model_path):
        print(Fore.YELLOW + "Training model first...")
        model, history = train_model()
    else:
        print(Fore.GREEN + "Model already exists.")

    # Launch Gradio interface
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "" #ur huggingface api token
    demo = create_integrated_interface()
    demo.launch(share=True)  # share=True creates a public link

if __name__ == "__main__":
    main()
