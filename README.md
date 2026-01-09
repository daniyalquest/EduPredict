# 🎓 EduPredict: Student Performance Analytics & Prediction

EduPredict is an end-to-end data science application designed to analyze student performance trends and predict overall academic outcomes using Machine Learning. It features a robust dashboard, a secure authentication system, and a real-time AI predictor.

## 🚀 Features

* **Secure Authentication**: User login/registration system with password hashing (bcrypt).
* **AI Score Predictor**: Uses a trained Machine Learning model to predict a student's overall score based on 13 different features.
* **Dynamic Analytics Dashboard**: 10 interactive visualizations covering:
    * Score distributions and correlations.
    * Impact of attendance and study hours.
    * Socio-economic factors (Parent Education, Travel Time).
* **Database Explorer**: Live view of records stored in MongoDB Atlas with search and CSV export capabilities.

## 🛠️ Tech Stack

* **Frontend**: Streamlit
* **Backend**: Python
* **Database**: MongoDB Atlas
* **Machine Learning**: Scikit-Learn, Joblib
* **Visualizations**: Seaborn, Matplotlib, Tableau
* **Security**: Bcrypt

## 📂 Project Structure

- `app.py`: The main Streamlit application.
- `prepare_data.py`: Data cleaning and feature engineering script.
- `train_model.py`: Script to train the ML model and save encoders.
- `test_model.py`: Performance evaluation script.
- `sync_db.py`: Script to sync local JSON data to MongoDB Atlas.
- `student_model.pkl`: The trained Random Forest/Regression model.
- `encoders.pkl`: Serialized LabelEncoders for categorical data.

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repo-link>
   cd EduPredict
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```