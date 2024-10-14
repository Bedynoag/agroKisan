import os
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from flask import Flask, redirect, url_for, request, render_template, jsonify
from flask import jsonify
import requests
from bs4 import BeautifulSoup as bs
from PIL import Image
from torchvision import transforms
import warnings
from werkzeug.utils import secure_filename
import torch.nn.functional as F
from io import BytesIO

app = Flask(__name__, static_folder=r"static")

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

device = get_default_device()

def to_device(data, device):
    """Move tensor(s) or model to chosen device (GPU/CPU)"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

# for calculating the accuracy
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
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
        
    def forward(self, x): # x is the loaded batch
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out   
    
loaded_model = CNN_NeuralNet(3, 38)
loaded_model.load_state_dict(torch.load('best_model.pth', map_location=device))  # Use the device directly
loaded_model = to_device(loaded_model, device)  # Move model to the appropriate device
loaded_model.eval()

# Load other necessary data
data = pd.DataFrame(pd.read_csv("final.csv"))
model = pickle.load(open(r"RandomForest.pkl", "rb"))
area = pd.read_csv(r"final_data.csv")
commodity = pd.read_csv(r"commodities1.csv")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictor')
def predictor():
    return render_template('predictor.html')

def now_final(state_name, district_name, market, commodity_name, trend, Datefrom, DateTo):
    commodity_code = commodity[commodity['Commodities'] == commodity_name]['code'].unique()[0]
    state_short_name = area[area['State'] == state_name]['State_code'].unique()[0]
    district_code = area[area['District'] == district_name]['District_code'].unique()[0]
    market_code = area[area['Market'] == market]['Market_code'].unique()[0]
    date_from = Datefrom
    date_to = DateTo
    commodity_name = commodity
    state_full_name = state_name
    district_full_name = district_name
    market_full_name = market

    r = requests.get(f"https://agmarknet.gov.in/SearchCmmMkt.aspx?Tx_Commodity={commodity_code}&Tx_State={state_short_name}&Tx_District={district_code}&Tx_Market={market_code}&DateFrom={date_from}&DateTo={date_to}&Fr_Date={date_from}&To_Date={date_to}&Tx_Trend={trend}&Tx_CommodityHead={commodity_name}&Tx_StateHead={state_full_name}&Tx_DistrictHead={district_full_name}&Tx_MarketHead={market_full_name}")
    soup = bs(r.text, "html.parser")
    title = soup.find("h4")

    tables = soup.find_all("table", class_="tableagmark_new")
    for tn in range(len(tables)):
        table = tables[tn]

        # preinit list of lists
        rows = table.findAll("tr")
        row_lengths = [len(r.findAll(['th', 'td'])) for r in rows]
        ncols = max(row_lengths)
        nrows = len(rows)
        data = []

        print(ncols, nrows)
        for i in range(nrows):
            rowD = []
            for j in range(ncols):
                rowD.append('')
            data.append(rowD)

        # process html
        for i in range(len(rows)):
            row = rows[i]
            cells = row.findAll(["td", "th"])
            j = 0  # Column index for data list

            if trend == "2":
                for cell in cells:
                    rowspan = int(cell.get('rowspan', 1))
                    colspan = int(cell.get('colspan', 1))
                    cell_text = cell.text.strip()

                    while data[i][j]:
                        j += 1

                    for r in range(rowspan):
                        for c in range(colspan):
                            data[i + r][j + c] = cell_text

                    j += colspan
            if trend == "0":
                if (i <= 50):
                    for cell in cells:
                        rowspan = int(cell.get('rowspan', 1))
                        colspan = int(cell.get('colspan', 1))
                        cell_text = cell.text.strip()

                        while data[i][j]:
                            j += 1

                        for r in range(rowspan):
                            for c in range(colspan):
                                data[i + r][j + c] = cell_text

                        j += colspan

        #     print(data)
        df = pd.DataFrame(data)
        df.columns = df.iloc[0]
        df = df[1:]
        if trend == "0":
            df = df.drop(df.index[-2:], axis=0)

        if df.empty:
            df.loc[0] = "No Data Found"
        if trend == "0":
            df.drop(columns={"Sl no."}, inplace=True)
        # df.to_csv("pta.csv",index=False)
        return (df, title.text)

@app.route('/market')
def market():
    state1 = sorted(area['State'].unique().astype(str))
    # print(states)  # Add this line to print the states

    area["District_state"] = area["State"] + "_" + area["District"]
    district1 = sorted(area['District_state'].unique().astype(str))
    # print(district1)
    area["market_district"] = area["State"]+"_" +area['District'] + "_" + area['Market']
    markets = sorted(area['market_district'].unique().astype(str))
    commodities= commodity["Commodities"].unique().astype(str)
    return render_template('market.html', states=state1, districts=district1, commodities=commodities, markets = markets)

@app.route('/price',methods=['POST'])
def price():
    state_name = request.form.get('state1')
    district_name = request.form.get('district1')
    market = request.form.get('Market')
    commodity_name = request.form.get('commodity')
    trend = request.form.get('Price/Arrival')
    Datefrom = request.form.get('from')
    DateTo = request.form.get('To')
    print(trend)
    final_data,heading=now_final(state_name,district_name,market,commodity_name,trend,Datefrom,DateTo)
    table_data = final_data.to_dict(orient='records')

    return jsonify(table_data)


disease_classes = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']


@app.route('/predict', methods=['POST'])
def predict():
    pH = float(request.form.get('ph'))
    rainfall = float(request.form.get('rainfall'))
    temperature = float(request.form.get('temperature'))
    nitrogen = int(request.form.get('nitrogen'))
    phosphorus = int(request.form.get('phosphorus'))
    potassium = int(request.form.get('potassium'))
    humidity = float(request.form.get('humidity'))
    output = model.predict([[nitrogen, phosphorus, potassium, temperature, humidity, pH, rainfall]]),

    return ("{} Crop can be Grown".format(str(output[0])))

def about_disease(filtered_df, column):
    cause_values = filtered_df[column]
    f = ", ".join(cause_values)
    return f.rstrip(", ") # Remove trailing comma and spaces


transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize the image to the input size of the model
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize as per the model's requirements
])

def predict_image(img, model):
    xb = to_device(img.unsqueeze(0), device)
    yb = model(xb)
    # Print the model's output (probabilities or logits)
    print("Model output:", yb)
    _, preds = torch.max(yb, dim=1)
    print("Predicted class index:", preds[0].item())
    return disease_classes[preds[0].item()]


@app.route('/disease', methods=['GET'])
def disease():
    # Main page
    return render_template('disease.html')

@app.route('/disease_pred', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        
        # Open the image file directly in memory using BytesIO
        img = Image.open(BytesIO(f.read()))
        
        # Apply transformations
        img = transform(img)
        
        # Make prediction
        preds = predict_image(img, loaded_model)

        # Assuming that the output is a tensor of probabilities
        filtered_df = data[data['Type'] == preds]

        symptoms = about_disease(filtered_df, 'Symptoms')
        cause = about_disease(filtered_df, 'Cause')
        prevention = about_disease(filtered_df, 'Prevention')

        response = {
            'disease': preds,
            'cause': cause,
            'symptoms': symptoms,
            'prevention': prevention
        }

        return jsonify(response)

@app.route('/rainfall', methods = ['GET', 'POST'])
def rainfall():
    return render_template('rainfall.html')

if __name__ == '__main__':
    app.run()