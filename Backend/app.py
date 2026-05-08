from flask import Flask, request, render_template, jsonify
import os
import pickle
import json
import numpy as np

# ---------------- Load Model ----------------

# ---------------- Load Columns ----------------
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "banglore_home_price_model.pickle")
model = pickle.load(open(model_path, "rb"))

columns_path = os.path.join(base_dir, "columns.json")
with open(columns_path, "r") as f:
    cols = json.load(f)

# Handle both possible keys in JSON
if "data_columns" in cols:
    data_columns = cols["data_columns"]
elif "columns" in cols:
    data_columns = cols["columns"]
else:
    data_columns = cols  # plain list

# First 3 are numerical, rest are locations
locations = data_columns[3:]

# Debug print
print("DEBUG Data Columns (first 20):", data_columns[:20])
print("DEBUG Extracted Locations (first 20):", locations[:20])

# ---------------- Flask App ----------------
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", locations=locations)

@app.route("/predict", methods=["POST"])
def predict():
    area = float(request.form["area"])
    bhk = int(request.form["bhk"])
    bath = int(request.form["bath"])
    location = request.form["location"].lower()

    # Create zero array for all features
    x = np.zeros(len(data_columns))

    # Fill numerical values
    if "total_sqft" in data_columns:
        x[data_columns.index("total_sqft")] = area
    if "bhk" in data_columns:
        x[data_columns.index("bhk")] = bhk
    if "bath" in data_columns:
        x[data_columns.index("bath")] = bath

    # Fill location one-hot
    if location in data_columns:
        loc_index = data_columns.index(location)
        x[loc_index] = 1

    # Make prediction
    prediction = model.predict([x])[0]

    return render_template(
        "index.html",
        locations=locations,
        prediction_text=f"Predicted Price: ₹ {round(prediction, 2)} Lakhs"
    )

# New API route to get all locations (for Postman / frontend)
@app.route("/locations", methods=["GET"])
def get_locations():
    return jsonify({"locations": locations})

# ---------------- Run ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
