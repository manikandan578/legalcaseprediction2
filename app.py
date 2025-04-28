from flask import Flask, render_template, request, redirect, url_for, send_file, flash, session
import pandas as pd
import joblib
import os
import random
import string
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from reportlab.pdfgen import canvas
from nltk.tokenize import sent_tokenize
from twilio.rest import Client
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Initialize app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'default_secret_key')  # Use environment variable for security
app.config['SESSION_TYPE'] = 'filesystem'

# Email and Twilio setup
EMAIL_ADDRESS = os.getenv('EMAIL_ADDRESS')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')
TWILIO_SID = os.getenv('TWILIO_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')

# Load dataset and train models (global loading to avoid redundancy)
df = pd.read_csv('land_cases_dataset_10000_with_favor.csv')
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Case_Text'])
y_case_type = df['Case_Type']
y_favor_result = df['Favor_Result']

model_case_type = LogisticRegression()
model_favor_result = LogisticRegression()
model_case_type.fit(X, y_case_type)
model_favor_result.fit(X, y_favor_result)

# Save models for later use
joblib.dump((vectorizer, model_case_type, model_favor_result), 'model.pkl')

# Helper Functions
def generate_summary(text, lines=3):
    sentences = sent_tokenize(text)
    return ' '.join(sentences[:lines])

def send_email(receiver_email, subject, body):
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        print("Email credentials are not set.")
        return
    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)

def send_sms(phone_number, message):
    if not TWILIO_SID or not TWILIO_AUTH_TOKEN:
        print("Twilio credentials are not set.")
        return
    client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
    client.messages.create(
        body=message,
        from_=TWILIO_PHONE_NUMBER,
        to=phone_number
    )

def generate_pdf(name, case_type, favor_result, confidence, summary):
    filename = f"{name}_case_report.pdf"
    c = canvas.Canvas(filename)
    c.drawString(100, 750, f"Name: {name}")
    c.drawString(100, 730, f"Predicted Case Type: {case_type}")
    c.drawString(100, 710, f"Predicted Favor Result: {favor_result}")
    c.drawString(100, 690, f"Prediction Confidence: {confidence:.2f}%")
    c.drawString(100, 670, f"Summary: {summary}")
    c.save()
    return filename

# Routes
@app.route('/')
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def do_login():
    username = request.form['username']
    password = request.form['password']

    if username == 'manikandan' and password == '6379039339':
        return redirect(url_for('predict'))
    else:
        flash('Invalid Username or Password')
        return redirect(url_for('login'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        name = request.form['name']
        case_text = request.form['case_text']
        email = request.form['email']
        phone = request.form['phone']

        # Sanitize inputs
        name = name.strip()
        case_text = case_text.strip()

        # Load model
        vectorizer, model_case_type, model_favor_result = joblib.load('model.pkl')

        X_input = vectorizer.transform([case_text])
        case_type_pred = model_case_type.predict(X_input)[0]
        favor_result_pred = model_favor_result.predict(X_input)[0]

        case_type_confidence = max(model_case_type.predict_proba(X_input)[0]) * 100
        favor_result_confidence = max(model_favor_result.predict_proba(X_input)[0]) * 100

        summary = generate_summary(case_text)

        # Save details in session
        session['report'] = {
            'name': name,
            'case_type': case_type_pred,
            'favor_result': favor_result_pred,
            'confidence': (case_type_confidence + favor_result_confidence) / 2,
            'summary': summary,
            'email': email,
            'phone': phone
        }

        return render_template('predict.html', prediction=session['report'])

    return render_template('predict.html', prediction=None)

@app.route('/send', methods=['POST'])
def send():
    report = session.get('report')
    if report:
        email_body = f"""
        Hello {report['name']},

        Predicted Case Type: {report['case_type']}
        Predicted Favor Result: {report['favor_result']}
        Prediction Confidence: {report['confidence']:.2f}%
        Summary: {report['summary']}
        """
        send_email(report['email'], "Legal Case Prediction Report", email_body)
        send_sms(report['phone'], f"Hi {report['name']}, your case prediction result has been sent to your email.")

        flash('Email and SMS sent successfully!')
    return redirect(url_for('predict'))

@app.route('/download')
def download():
    report = session.get('report')
    if report:
        filename = generate_pdf(report['name'], report['case_type'], report['favor_result'], report['confidence'], report['summary'])
        return send_file(filename, as_attachment=True)
    return redirect(url_for('predict'))

if __name__ == '__main__':
    app.run(debug=True)
