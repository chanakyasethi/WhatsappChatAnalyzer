import boto3
import pandas as pd

def get_sentiment(text):

    comprehend = boto3.client(service_name='comprehend', region_name='ap-south-1')

    sentiment_output = comprehend.detect_sentiment(Text=text, LanguageCode='en')

    return sentiment_output.get('Sentiment')