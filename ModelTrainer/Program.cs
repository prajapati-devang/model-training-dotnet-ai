using Microsoft.ML;
using ModelTrainer;

var context = new MLContext();

// 1. Prepare Data (In a real app, load this from a .csv file)
var data = new[] {
    new SentimentData { Text = "This is excellent!", Sentiment = true },
    new SentimentData { Text = "I am very disappointed.", Sentiment = false },
    new SentimentData { Text = "Highly recommended!", Sentiment = true },
    new SentimentData { Text = "Worst experience ever.", Sentiment = false }
};
var trainingData = context.Data.LoadFromEnumerable(data);

// 2. Build Pipeline: Text -> Numbers -> Algorithm
var pipeline = context.Transforms.Text.FeaturizeText("Features",
    nameof(SentimentData.Text))
    .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression());

// 3. Train
var model = pipeline.Fit(trainingData);

// 4. Save the model as a .zip file for use this model for predict the Sentiment analysis.
context.Model.Save(model, trainingData.Schema, "SentimentModel.zip");

ZipFileCopier.CopyZipToBlazorApp("SentimentModel.zip");

Console.WriteLine("Model saved successfully to SentimentModel.zip");