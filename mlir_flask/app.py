from flask import Flask, render_template, request, jsonify
import services

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():# Display form
    return render_template("predict_form.html", result=None)

@app.route("/predict", methods=["POST"])
def predict_view():# Process form submission
    try:                                             
        sl = float(request.form.get("sepal_length", ""))
        sw = float(request.form.get("sepal_width", ""))
        pl = float(request.form.get("petal_length", ""))
        pw = float(request.form.get("petal_width", ""))
    except (ValueError, TypeError):
        return render_template("predict_form.html", result={"error": "Invalid input"}, form_values=request.form)

    result = services.predict_iris([sl, sw, pl, pw])
    return render_template("predict_form.html", result=result, form_values=request.form)

@app.route("/api/predict", methods=["POST"]) # API endpoint
def predict_api():# Handle JSON or form data
    if request.is_json:
        payload = request.get_json()
    else:
        payload = request.form.to_dict() # Convert form data to dict

    required = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    missing = [k for k in required if k not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400 # Bad Request

    try:
        features = [float(payload[k]) for k in required]# Convert to float
    except ValueError:
        return jsonify({"error": "All features must be numeric."}), 400 

    result = services.predict_iris(features)# Make prediction
    return jsonify(result)# Return JSON response

if __name__ == "__main__":
    app.run(debug=True)
