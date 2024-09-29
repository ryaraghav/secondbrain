from langchain_community.document_loaders import YoutubeLoader, TextLoader

def load_youtube_data(url):
    return YoutubeLoader.from_youtube_url(url, add_video_info=False).load()

def load_text_data(file_path):
    return TextLoader(file_path).load()