using Microsoft.ML.Data;

public class SentimentData
{
    [LoadColumn(0)] public string Text { get; set; } = default!;
    [LoadColumn(1), ColumnName("Label")] public bool Sentiment { get; set; }
}
