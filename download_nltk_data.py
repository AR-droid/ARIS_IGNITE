import nltk

# Download required NLTK data
def download_nltk_data():
    print("Downloading NLTK data...")
    nltk.download('punkt', quiet=False)
    nltk.download('stopwords', quiet=False)
    nltk.download('punkt_tab', quiet=False)
    print("NLTK data download complete!")

if __name__ == "__main__":
    download_nltk_data()
