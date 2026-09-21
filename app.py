from flask import Flask, render_template, request, redirect, url_for, session, send_file
import joblib
import pandas as pd
import json
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
import os
from dotenv import load_dotenv

app = Flask(__name__)

# =========================================================
# SECRET KEY
# =========================================================

load_dotenv()

app.secret_key = os.environ.get("SECRET_KEY")


# =========================================================
# LOAD TRAINED MACHINE LEARNING MODEL
# =========================================================

model_package = joblib.load("heart_disease_model.pkl")

model = model_package["model"]
feature_names = model_package["feature_names"]
encoders = model_package["encoders"]

print("Model loaded successfully!")
print("Features expected by model:")
print(feature_names)


# =========================================================
# PREDICTION HISTORY
# =========================================================

HISTORY_FILE = "prediction_history.json"


def load_history():

    try:

        with open(HISTORY_FILE, "r") as file:
            return json.load(file)

    except (FileNotFoundError, json.JSONDecodeError):

        return []


def save_history(history):

    with open(HISTORY_FILE, "w") as file:

        json.dump(
            history,
            file,
            indent=4
        )


# =========================================================
# LOGIN
# =========================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form.get("email")
        password = request.form.get("password")

        # Temporary login credentials
        # Later this will be connected to RDS.

        if email == os.environ.get("ADMIN_EMAIL") and password == os.environ.get("ADMIN_PASSWORD"):

            session["logged_in"] = True
            session["user_name"] = "Admin"

            return redirect(
                url_for("dashboard")
            )

        else:

            return render_template(
                "login.html",
                error="Invalid email or password"
            )

    return render_template("login.html")


# =========================================================
# DASHBOARD
# =========================================================

@app.route("/dashboard")
def dashboard():

    if not session.get("logged_in"):

        return redirect(
            url_for("login")
        )

    history = load_history()

    total_predictions = len(history)

    # PDF report count
    reports_folder = "reports"

    if os.path.exists(reports_folder):

        total_reports = len([
            file
            for file in os.listdir(reports_folder)
            if file.endswith(".pdf")
        ])

    else:

        total_reports = 0

    return render_template(

        "dashboard.html",

        total_predictions=total_predictions,

        total_history=total_predictions,

        total_reports=total_reports

    )


# =========================================================
# PREDICTION HISTORY
# =========================================================

@app.route("/history")
def history():

    if not session.get("logged_in"):

        return redirect(
            url_for("login")
        )

    prediction_history = load_history()

    return render_template(

        "history.html",

        history=prediction_history

    )


# =========================================================
# LOGOUT
# =========================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("login")
    )


# =========================================================
# HOME / NEW PREDICTION
# =========================================================

@app.route("/")
def home():

    if not session.get("logged_in"):

        return redirect(
            url_for("login")
        )

    return render_template(

        "index.html",

        feature_names=feature_names,

        encoders=encoders

    )


# =========================================================
# PREDICTION
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    if not session.get("logged_in"):

        return redirect(
            url_for("login")
        )

    try:

        patient_name = request.form.get(
            "patient_name"
        )

        input_data = {}


        # ---------------------------------------------
        # GET PATIENT DATA
        # ---------------------------------------------

        for feature in feature_names:

            value = request.form.get(feature)

            if feature in encoders:

                value = encoders[
                    feature
                ].transform([value])[0]

            else:

                value = float(value)

            input_data[feature] = value


        # ---------------------------------------------
        # CREATE DATAFRAME
        # ---------------------------------------------

        patient_data = pd.DataFrame(

            [input_data],

            columns=feature_names

        )


        # ---------------------------------------------
        # MODEL PREDICTION
        # ---------------------------------------------

        prediction = model.predict(
            patient_data
        )[0]


        probabilities = model.predict_proba(
            patient_data
        )[0]


        confidence = round(

            max(probabilities) * 100,

            2

        )


        # ---------------------------------------------
        # RESULT
        # ---------------------------------------------

        if prediction == 1:

            result = "Heart Disease Detected"

            status = "high"

        else:

            result = "No Heart Disease Detected"

            status = "low"


        # ---------------------------------------------
        # SAVE HISTORY
        # ---------------------------------------------

        prediction_id = len(
            load_history()
        ) + 1

        prediction_date = datetime.now().strftime(
            "%d-%m-%Y %H:%M"
        )


        history = load_history()


        history.append({

            "id": prediction_id,

            "patient_name": patient_name,

            "result": result,

            "confidence": confidence,

            "status": status,

            "date": prediction_date

        })


        save_history(history)


        # ---------------------------------------------
        # RESULT PAGE
        # ---------------------------------------------

        return render_template(

            "result.html",

            patient_name=patient_name,

            result=result,

            confidence=confidence,

            status=status,

            prediction_id=prediction_id

        )


    except Exception as e:

        return render_template(

            "result.html",

            patient_name="Patient",

            result="Unable to make prediction",

            confidence=0,

            status="error",

            error=str(e)

        )


# =========================================================
# GENERATE PDF REPORT
# =========================================================

@app.route("/generate_report/<int:prediction_id>")
def generate_report(prediction_id):

    if not session.get("logged_in"):

        return redirect(
            url_for("login")
        )


    history = load_history()


    # Find prediction
    selected_prediction = None

    for record in history:

        if record.get("id") == prediction_id:

            selected_prediction = record

            break


    if selected_prediction is None:

        return "Prediction not found"


    # Create reports folder
    reports_folder = "reports"

    os.makedirs(
        reports_folder,
        exist_ok=True
    )


    # Safe patient name
    patient_name = selected_prediction[
        "patient_name"
    ]

    safe_name = "".join(

        character
        for character in patient_name
        if character.isalnum() or character in " _-"

    ).strip()


    if not safe_name:

        safe_name = "Patient"


    filename = (

        f"{safe_name}_HeartCare_Report_"
        f"{prediction_id}.pdf"

    )


    filepath = os.path.join(

        reports_folder,

        filename

    )
    


    # =====================================================
    # PDF DOCUMENT
    # =====================================================

    document = SimpleDocTemplate(

        filepath,

        pagesize=A4,

        rightMargin=45,

        leftMargin=45,

        topMargin=45,

        bottomMargin=45

    )


    styles = getSampleStyleSheet()


    title_style = ParagraphStyle(

        "TitleStyle",

        parent=styles["Title"],

        fontSize=24,

        alignment=TA_CENTER,

        spaceAfter=10

    )


    subtitle_style = ParagraphStyle(

        "SubtitleStyle",

        parent=styles["Normal"],

        fontSize=11,

        alignment=TA_CENTER,

        textColor=colors.grey,

        spaceAfter=25

    )


    heading_style = ParagraphStyle(

        "HeadingStyle",

        parent=styles["Heading2"],

        fontSize=15,

        spaceBefore=12,

        spaceAfter=10

    )


    normal_style = ParagraphStyle(

        "NormalStyle",

        parent=styles["Normal"],

        fontSize=10,

        leading=15

    )


    story = []


    # Title
    story.append(

        Paragraph(
            "HeartCare AI",
            title_style
        )

    )


    story.append(

        Paragraph(

            "AI-Based Heart Disease Prediction Report",

            subtitle_style

        )

    )


    # Patient Information
    story.append(

        Paragraph(
            "Patient Information",
            heading_style
        )

    )


    patient_table = Table([

        [
            "Patient Name",
            selected_prediction["patient_name"]
        ],

        [
            "Prediction ID",
            str(selected_prediction["id"])
        ],

        [
            "Date",
            selected_prediction["date"]
        ]

    ], colWidths=[150, 330])


    patient_table.setStyle(

        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.lightgrey
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "FONTNAME",
                (0, 0),
                (-1, -1),
                "Helvetica"
            ),

            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                10
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                8
            )

        ])

    )


    story.append(
        patient_table
    )

    story.append(
        Spacer(1, 20)
    )


    # Prediction Result
    story.append(

        Paragraph(
            "Prediction Result",
            heading_style
        )

    )


    result_table = Table([

        [
            "Prediction",
            selected_prediction["result"]
        ],

        [
            "Model Confidence",
            str(
                selected_prediction["confidence"]
            ) + "%"
        ]

    ], colWidths=[150, 330])


    result_table.setStyle(

        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.lightgrey
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                10
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                8
            )

        ])

    )


    story.append(
        result_table
    )


    story.append(
        Spacer(1, 25)
    )


    # Disclaimer
    story.append(

        Paragraph(
            "Important Notice",
            heading_style
        )

    )


    story.append(

        Paragraph(

            "HeartCare AI provides machine-learning based "
            "predictions for educational and awareness "
            "purposes only. This report is not a medical "
            "diagnosis and should not replace professional "
            "medical advice. Please consult a qualified "
            "healthcare professional for medical concerns.",

            normal_style

        )

    )


    story.append(
        Spacer(1, 20)
    )


    story.append(

        Paragraph(

            "Generated by HeartCare AI",

            subtitle_style

        )

    )


    # Build PDF
    document.build(story)


    # Send PDF to browser
    return send_file(

        filepath,

        as_attachment=True,

        download_name=filename

    )
# =========================================================
# MEDICAL REPORTS PAGE
# =========================================================

@app.route("/reports")
def reports():

    if not session.get("logged_in"):
        return redirect(url_for("login"))

    prediction_history = load_history()

    return render_template(
        "reports.html",
        history=prediction_history
    )

# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(debug=True)