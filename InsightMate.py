from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize

# Download NLTK tokenizer
# nltk.download("punkt")
nltk.download('punkt_tab')

class InsightChatbot:
    def __init__(self):
        # Initialize transformers pipeline for summarization and sentiment analysis
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.topics = ["Productivity", "Career Guidance", "Life Hacks", "General Advice"]

    def extract_insights(self, text, topic):
        """
        Extract specific insights related to the requested topic.
        """
        sentences = sent_tokenize(text)
        relevant_sentences = []
        
        for sentence in sentences:
            classification = self.classifier(sentence, candidate_labels=self.topics)
            if classification["labels"][0] == topic:
                relevant_sentences.append(sentence)
        
        if relevant_sentences:
            insights = " ".join(relevant_sentences)
            summarized = self.summarizer(insights, max_length=100, min_length=25, do_sample=False)
            return summarized[0]["summary_text"]
        else:
            return "No insights found for the requested topic."

    def chat(self):
        """
        Engage with the user to analyze stories and provide insights.
        """
        print("Hello! I can extract insights from stories or long text. Provide the text:")
        text = input("\nEnter your text (or type 'exit' to quit):\n")
        while text.lower() != "exit":
         print("\nWhat kind of insights do you want? (Options: Productivity, Career Guidance, Life Hacks)")
         topic = input("Enter your choice: ").strip().capitalize()  # Strip spaces and ensure proper capitalization
         if topic in self.topics:
          print("\nAnalyzing for insights on:", topic)
          insights = self.extract_insights(text, topic)
          print("\nInsights:",insights)
         else:
          print("Invalid topic. Please choose from the given options.")
         text = input("\nEnter another text (or type 'exit' to quit):\n")


if __name__ == "__main__":
    chatbot = InsightChatbot()
    chatbot.chat()
