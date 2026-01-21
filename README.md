House Price Prediction AI


Introduction


This project is an AI-powered real estate valuation tool designed to estimate property prices based on various physical and situational features. By leveraging Machine Learning, the application analyzes historical housing data to provide users with accurate market estimates, helping buyers and sellers make informed decisions.

Core Features
Intelligent Valuation: Predicts house prices based on area, bedrooms, bathrooms, floors, and furnishing status.

Advanced AI Model: Utilizes a Random Forest Regressor for robust and accurate numerical predictions.

Comprehensive Input Analysis: Factors in premium features like main road access, guest rooms, basements, and air conditioning.

Interactive Web Interface: A clean, modern, and fully responsive user interface built with HTML5 and CSS3.

Instant Feedback: Users receive real-time price estimations with built-in data validation.

Technologies Used
Frontend: HTML5, CSS3 (Grid & Flexbox), JavaScript (ES6+).

Backend: Python (Flask/Django assumed for model deployment).

Machine Learning: Scikit-learn (Random Forest Regressor).

Data Processing: Pandas, NumPy.

Dataset: Housing.csv (Commonly used for real estate regression tasks).

How the AI Model Works
The prediction engine is built on the Random Forest algorithm, which works by:

Constructing multiple decision trees during training.

Analyzing features such as total area, number of rooms, and amenities.

Outputting the average prediction (mean) from all individual trees to improve accuracy and prevent overfitting.

Installation and Setup
Clone the repository: git clone https://github.com/YuluWusu/real-estate-valuation-tool.git

Install dependencies: pip install flask pandas scikit-learn numpy

Ensure 'housing.csv' is in the project directory.

Run the application: python app.py

Access the tool: Open http://127.0.0.1:5000 in your browser.

Project Structure
index.html: The main landing page and prediction form.

style.css: Modern styling with gradient backgrounds and responsive layouts.

model.py: Script for training and saving the Random Forest model.

app.py: The web server handling requests and model inference.
