import pandas as pd
from flask import Flask, render_template, request
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer
import os

app = Flask(__name__)

folder = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(folder, "kdramas_expanded.csv"), encoding="cp1252")
df["Genre"] = df["Genre"].fillna("").apply(
    lambda x: [g.strip() for g in x.split(",") if g.strip()])
# Convert rating and year into numbers
df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
# Convert genres into numerical values
encoder = MultiLabelBinarizer()
x = encoder.fit_transform(df["Genre"])

#  KNN model
model = NearestNeighbors(metric="manhattan")
model.fit(x)

# Get all genres and years
genres = sorted(encoder.classes_)
years = sorted(df["Year"].dropna().unique(), reverse=True)

@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []
    selected_genres = []
    selected_rating = ""
    selected_year = ""
    message = ""
    if request.method == "POST":

        # Get the user's selections
        selected_genres = request.form.getlist("genres")
        selected_rating = request.form.get("rating")
        selected_year = request.form.get("year")

        # Check whether the user selected anything
        if not selected_genres and not selected_rating and not selected_year:
            message = "Please select at least one filter!"

        else:
            # Find dramas using KNN when genres are selected
            if selected_genres:
                user_input = encoder.transform([selected_genres])
                distance, neighbours = model.kneighbors(
                    user_input, n_neighbors=len(df)
                )
                results = df.iloc[neighbours[0]]

                # Keep dramas that have all selected genres
                results = results[
                    results["Genre"].apply(
                        lambda x: all(g in x for g in selected_genres) )]
            else:
                results = df.copy()
 # Arrange dramas by highest rating
                results = results.sort_values( by="Rating", ascending=False )
            # Filter dramas by rating
            if selected_rating:
                results = results[
                    results["Rating"] >= float(selected_rating)]

            # Filter dramas by year
            if selected_year:
                results = results[
                    results["Year"] == int(selected_year) ]

            # Display the first five dramas
            recommendations = results["Drama"].head(5).tolist()
            if len(recommendations) == 0:
                message = "No dramas found. Try different filters!"

    return render_template(
        "index.html",
        genres=genres,
        years=years,
        recommendations=recommendations,
        selected_genres=selected_genres,
        selected_rating=selected_rating,
        selected_year=selected_year,
        message=message
    )

if __name__ == "__main__": app.run(debug=True)