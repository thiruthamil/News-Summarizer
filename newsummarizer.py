import requests
import matplotlib.pyplot as plt
from transformers import BartForConditionalGeneration, BartTokenizer

# API Key for NewsAPI (Replace with your API key)
API_KEY = "9c048578b34647e6aa91fb609d76649d"

# Initialize BART model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

def fetch_news_by_category(category, country='us', page_size=5):
    """
    Fetch news articles for a specific category using NewsAPI.
    """
    url = f"https://newsapi.org/v2/top-headlines?category={category}&country={country}&pageSize={page_size}&apiKey={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        articles = data.get("articles", [])
        return articles
    else:
        print(f"Error fetching news: {response.status_code}")
        return []

def generate_summary_with_url(text, url, max_length=250, min_length=100):
    """
    Generate abstractive summary using the BART model and include the source URL.
    """
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    # Split the summary into sentences and limit to 10 sentences
    sentences = summary.split('. ')[:10]
    summary_with_url = '. '.join(sentences) + f". [Read more here]({url})"
    return summary_with_url

def main():
    print("Welcome to the Enhanced Personalized News Summarizer!")
    print("Choose a category to get the latest news:")
    categories = ["business", "entertainment", "general", "health", "science", "sports", "technology"]
    print(", ".join(categories))
    
    category = input("Enter your category of interest: ").lower()
    if category not in categories:
        print("Invalid category. Please choose a valid category from the list.")
        return
    
    print("\nFetching news articles...")
    articles = fetch_news_by_category(category)
    
    if not articles:
        print("No articles found or an error occurred while fetching news.")
        return
    
    for i, article in enumerate(articles):
        print(f"\nArticle {i+1}: {article['title']}")
        print(f"Source: {article['source']['name']}")
        print(f"Published At: {article['publishedAt']}")
        
        # Generate summary for the article content
        if article.get("content"):
            print("\nGenerating summary...")
            summary = generate_summary_with_url(article["content"], article["url"])
            print("\nSummary:")
            print(summary)
        else:
            print("No content available for summarization.")

if __name__ == "__main__":
    main()
    #python "d:\Thiruthamil\newsummarizer.py"