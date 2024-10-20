from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the models and encoders
label_encoder = joblib.load('genre_label_encoder.pkl')
model = joblib.load('revenue_prediction_model.pkl')
sc = joblib.load('scaler.pkl')  # Assuming there's a scaler used for input transformation

# Genre dictionary
genre_dict = {
    'Action': 0, 'Adventure': 1, 'Animation': 2, 'Comedy': 3, 'Crime': 4, 'Documentary': 5,
    'Drama': 6, 'Family': 7, 'Fantasy': 8, 'Foreign': 9, 'History': 10, 'Horror': 11,
    'Music': 12, 'Mystery': 13, 'Romance': 14, 'Science Fiction': 15, 'TV Movie': 16,
    'Thriller': 17, 'War': 18, 'Western': 19, None: 20
}

@app.route('/')
def index():
    return render_template('index.html', genres=genre_dict.keys())

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    budget = float(request.form['budget'])
    popularity = float(request.form['popularity'])
    runtime = float(request.form['runtime'])
    vote_average = float(request.form['vote_average'])
    vote_count = int(request.form['vote_count'])
    genre = request.form['genre']
    release_month = int(request.form['release_month'])
    release_week = int(request.form['release_week'])

    # Create input data frame
    input_data = {
        'budget': budget,
        'popularity': popularity,
        'runtime': runtime,
        'vote_average': vote_average,
        'vote_count': vote_count,
        'genre_names': genre,
        'release_month': release_month,
        'release_week': release_week
    }

    input_df = pd.DataFrame([input_data])

    # Transform genre using label encoder
    input_df['genre_names'] = label_encoder.transform(input_df['genre_names'])

    # Scale the input data
    input_df_scaled = sc.transform(input_df)

    # Predict revenue
    predicted_revenue = model.predict(input_df_scaled)

    return render_template('result.html', predicted_revenue=predicted_revenue[0])

if __name__ == '__main__':
    app.run(debug=True)
