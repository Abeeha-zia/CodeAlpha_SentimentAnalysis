
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Step 1: Load the review dataset
df = pd.read_csv("sentiment_reviews.csv")

# Step 2: Check the dataset
print("SENTIMENT ANALYSIS PROJECT")
print("\nFirst five reviews:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

# Step 3: Clean the review text
df["Review"] = df["Review"].fillna("").astype(str).str.strip()
df = df[df["Review"] != ""].copy()

# Step 4: Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Step 5: Calculate sentiment scores
df["Positive Score"] = df["Review"].apply(
    lambda text: analyzer.polarity_scores(text)["pos"]
)
df["Negative Score"] = df["Review"].apply(
    lambda text: analyzer.polarity_scores(text)["neg"]
)
df["Neutral Score"] = df["Review"].apply(
    lambda text: analyzer.polarity_scores(text)["neu"]
)
df["Compound Score"] = df["Review"].apply(
    lambda text: analyzer.polarity_scores(text)["compound"]
)

# Step 6: Classify each review
def classify_sentiment(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df["Sentiment"] = df["Compound Score"].apply(classify_sentiment)

# Step 7: Print each review with its result
print("\nREVIEW SENTIMENT RESULTS")
print(df[["Review_ID", "Review", "Compound Score", "Sentiment"]].to_string(index=False))

# Step 8: Count and calculate percentages
sentiment_counts = (
    df["Sentiment"]
    .value_counts()
    .reindex(["Positive", "Negative", "Neutral"], fill_value=0)
)

sentiment_percentages = (sentiment_counts / len(df) * 100).round(2)

print("\nSENTIMENT COUNTS")
print(sentiment_counts)

print("\nSENTIMENT PERCENTAGES")
print(sentiment_percentages)

# Step 9: Save the results
df.to_csv("sentiment_analysis_results.csv", index=False)

# Chart 1: Sentiment distribution
plt.figure(figsize=(8, 5))
sentiment_counts.plot(kind="bar", edgecolor="black")
plt.title("Sentiment Distribution of Book Reviews")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("sentiment_distribution.png", dpi=300)
plt.show()

# Chart 2: Sentiment percentages
plt.figure(figsize=(7, 7))
sentiment_counts.plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Percentage of Positive, Negative and Neutral Reviews")
plt.ylabel("")
plt.tight_layout()
plt.savefig("sentiment_percentages.png", dpi=300)
plt.show()

# Step 10: Save a short summary
with open("sentiment_summary.txt", "w", encoding="utf-8") as file:
    file.write("SENTIMENT ANALYSIS SUMMARY\n\n")
    file.write(f"Total reviews analyzed: {len(df)}\n\n")
    file.write("Sentiment counts:\n")
    file.write(sentiment_counts.to_string())
    file.write("\n\nSentiment percentages:\n")
    file.write(sentiment_percentages.to_string())

print("\nFiles saved successfully:")
print("- sentiment_analysis_results.csv")
print("- sentiment_distribution.png")
print("- sentiment_percentages.png")
print("- sentiment_summary.txt")
