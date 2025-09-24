# League of Legends Win Predictor 🎮

Machine learning project predicting match outcomes from champion draft and objectives.

## Features
- Trained on 51k EUW ranked games dataset
- Feature engineering: champion picks, bans, roles
- RandomForest + XGBoost models
- Comparison: draft-only vs draft + early objectives
- Interactive [Streamlit app](https://league-of-legends-prediction.streamlit.app/) for predictions

## How to run locally

- git clone https://github.com/NickHolden404/league-of-legends-prediction.git
- cd league-of-legends-prediction
- pip install -r requirements.txt
- streamlit run app.py

---

## 4. Push to GitHub
From your project folder:

# if not initialized
- git init
- git add .
- git commit -m "Initial commit - League of Legends predictor"
- git branch -M main
- git remote add origin https://github.com/NickHolden404/league-of-legends-prediction.git
- git push -u origin main
