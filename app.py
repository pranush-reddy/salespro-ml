from flask import Flask, render_template, request, redirect, url_for
import joblib
from pymongo import MongoClient
import pandas as pd
import os
from datetime import datetime 

app = Flask(__name__)

# Load the pre-trained model
model_path = os.path.join(os.path.dirname(__file__), 'bigmart_model')
model = joblib.load(model_path)

mongo_url = "mongodb+srv://pranush01:3sf6Zt5iO3ZTj8RR@portfolio.ya0h1nc.mongodb.net/formdata"
client = MongoClient(mongo_url)

# Access the database
db = client.get_database()

# Access the collection
collection = db.salespro

# Define mapping for Item_Type
item_type_mapping = {
    'Dairy': 0,
    'Soft Drinks': 1,
    'Meat': 2,
    'Fruits and Vegetables': 3,
    'Household': 4,
    'Baking Goods': 5,
    'Snack Foods': 6,
    'Frozen Foods': 7,
    'Breakfast': 8,
    'Health and Hygiene': 9,
    'Hard Drinks': 10,
    'Canned': 11,
    'Breads': 12,
    'Starchy Foods': 13,
    'Others': 14,
    'Seafood': 15
}  

# Mapping for Outlet_Location_Type
outlet_location_mapping = {
    'Tier 1': 0,
    'Tier 2': 1,
    'Tier 3': 2
}
outlet_location_mapping2 = {
    'Rural': 'Tier 1',
    'Urban': 'Tier 2',
    'Semi-Urban': 'Tier 3'
}

# Mapping for Outlet_Type
outlet_type_mapping = {
    'Supermarket Type1': 2,
    'Supermarket Type2': 3,
    'Supermarket Type3': 1,
    'Grocery Store': 0
}
# Mapping for Outlet_Identifier
outlet_identifier_mapping = {
    'OUT010': 0,
    'OUT013': 1,
    'OUT017': 2,
    'OUT018': 3,
    'OUT019': 4,
    'OUT027': 5,
    'OUT035': 6,
    'OUT045': 7,
    'OUT046': 8,
    'OUT049': 9
}

# Mapping for Outlet_Size
outlet_size_mapping = {
    'Small': 0,
    'Medium': 1,
    'High': 2
}

# Minimum value for item MRP
MIN_ITEM_MRP = 1
MAX_ITEM_MRP = 270
# Change this to your desired minimum value

# Define the route for the home page
@app.route('/')
def home():
    return render_template('main.html')

# Define the route for the tool page
@app.route('/index.html')
def show():
    return render_template('index.html')

@app.route('/aboutus.html')
def show_tool():
    return render_template('aboutus.html')

# Define the route for the form page
@app.route('/form.html')
def show_form():
    return render_template('form.html')

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Extract input data from the form
    item_type = request.form['item_type']
    item_mrp = float(request.form['item_mrp'])
    outlet_identifier = request.form['outlet_identifier']
    outlet_size = request.form['outlet_size']
    outlet_location_type = request.form['outlet_location_type']
    outlet_type = request.form['outlet_type']
    
    # Validate item MRP
    if item_mrp < MIN_ITEM_MRP:
        return "Invalid input" 
    if item_mrp > MAX_ITEM_MRP:
        return "<center><h1 style='margin-top: 2%; '>According to our data contraints <br>few products range is below your inputs</h1></center><br><center><h2>Maximum value is 300</h2></center>" 
    
    # Convert categorical values to numeric using mapping
    item_type_numeric = item_type_mapping.get(item_type, -1)  # Default to -1 if not found
    outlet_type_numeric = outlet_type_mapping.get(outlet_type, -1)  # Default to -1 if not found
    outlet_location_type_numeric = outlet_location_mapping.get(outlet_location_mapping2.get(outlet_location_type, -1), -1) 
    outlet_identifier_numeric = outlet_identifier_mapping.get(outlet_identifier, -1)  # Default to -1 if not found
    outlet_size_numeric = outlet_size_mapping.get(outlet_size, -1)# Default to -1 if not found

    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'Item_Type': [item_type_numeric],
        'Item_MRP': [item_mrp],
        'Outlet_Identifier': [outlet_identifier_numeric],
        'Outlet_Size': [outlet_size_numeric],
        'Outlet_Location_Type': [outlet_location_type_numeric],
        'Outlet_Type': [outlet_type_numeric],
    })

    # Make the prediction
    predicted_sales = int(model.predict(input_data))

    # Render the prediction result template with the predicted sales value
    return render_template('result.html', predicted_sales=predicted_sales)

@app.route('/feeddata', methods=['POST'])
def submit_form():
    # Retrieve form data
    first_name = request.form['first_name']
    last_name = request.form['last_name']
    email = request.form['email']
    message = request.form['message']

    collection.insert_one({
        'first_name': first_name,
        'last_name': last_name,
        'email': email,
        'message': message,
        'timestamp': datetime.utcnow()  # Adding timestamp
    })
    # Render the same page template
    return render_template('form.html')

if __name__ == '__main__':
    # Create an empty form_data.txt file if it doesn't exist
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
