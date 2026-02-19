"""
Sentiment analysis module using X.com (Twitter) and news sources.
Calculates sentiment scores for trading decisions.
"""

import tweepy
import feedparser
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from textblob import TextBlob
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class XComSentimentAnalyzer:
    """Analyzes sentiment from X.com/Twitter posts."""
    
    def __init__(self, api_key: str, api_secret: str, bearer_token: str):
        """
        Initialize X.com API client.
        
        Args:
            api_key: X API key
            api_secret: X API secret
            bearer_token: X bearer token
        """
        self.bearer_token = bearer_token
        self.api_key = api_key
        self.api_secret = api_secret
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the X.com API client."""
        try:
            if self.bearer_token:
                # Using v2 API
                self.client = tweepy.Client(bearer_token=self.bearer_token)
                logger.info("X.com API client initialized (v2)")
            else:
                logger.warning("X.com bearer token not configured")
        except Exception as e:
            logger.error(f"Error initializing X.com client: {e}")
    
    def fetch_crypto_mentions(self, keyword: str, hours: int = 24) -> List[Dict]:
        """
        Fetch recent crypto mentions from X.com.
        
        Args:
            keyword: Cryptocurrency keyword (e.g., 'Bitcoin', 'BTC')
            hours: Hours of history to search
            
        Returns:
            List of tweets with metadata
        """
        if not self.client:
            logger.warning("X.com client not initialized")
            return []
        
        try:
            # Calculate time range
            start_time = datetime.utcnow() - timedelta(hours=hours)
            
            # Search for recent tweets
            tweets = []
            search_query = f"{keyword} -is:retweet lang:en"
            
            for tweet in self.client.search_recent_tweets(
                query=search_query,
                start_time=start_time,
                max_results=100,
                tweet_fields=['created_at', 'public_metrics', 'author_id']
            ).data or []:
                tweets.append({
                    'text': tweet.text,
                    'created_at': tweet.created_at,
                    'likes': tweet.public_metrics['like_count'],
                    'retweets': tweet.public_metrics['retweet_count'],
                    'replies': tweet.public_metrics['reply_count']
                })
            
            logger.info(f"Fetched {len(tweets)} mentions for {keyword}")
            return tweets
            
        except Exception as e:
            logger.error(f"Error fetching X.com mentions for {keyword}: {e}")
            return []
    
    def analyze_sentiment(self, texts: List[str]) -> float:
        """
        Analyze sentiment of text samples.
        
        Args:
            texts: List of text strings
            
        Returns:
            Average sentiment score (-1 to 1)
        """
        if not texts:
            return 0.0
        
        sentiments = []
        for text in texts:
            try:
                blob = TextBlob(text)
                sentiments.append(blob.sentiment.polarity)
            except Exception as e:
                logger.warning(f"Error analyzing text sentiment: {e}")
        
        return np.mean(sentiments) if sentiments else 0.0
    
    def get_crypto_sentiment(self, symbol: str, hours: int = 24) -> Dict:
        """
        Get aggregated sentiment for a cryptocurrency.
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC', 'ETH')
            hours: Hours of history to analyze
            
        Returns:
            Sentiment metrics dictionary
        """
        keyword = symbol.replace('-USD', '').upper()
        tweets = self.fetch_crypto_mentions(keyword, hours)
        
        if not tweets:
            return {
                'symbol': symbol,
                'sentiment_score': 0.0,
                'mention_count': 0,
                'engagement_score': 0.0,
                'timestamp': datetime.now()
            }
        
        # Calculate sentiment
        texts = [t['text'] for t in tweets]
        sentiment_score = self.analyze_sentiment(texts)
        
        # Calculate engagement
        total_engagement = sum(
            t['likes'] + t['retweets'] * 2 + t['replies']
            for t in tweets
        )
        avg_engagement = total_engagement / len(tweets)
        
        # Normalize engagement score to -1 to 1
        engagement_score = np.tanh(avg_engagement / 100)
        
        return {
            'symbol': symbol,
            'sentiment_score': sentiment_score,
            'mention_count': len(tweets),
            'engagement_score': engagement_score,
            'total_engagement': total_engagement,
            'timestamp': datetime.now()
        }


class NewsSentimentAnalyzer:
    """Analyzes sentiment from news sources."""
    
    def __init__(self):
        """Initialize news analyzer."""
        self.rss_feeds = [
            'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'https://feeds.bloomberg.com/markets/news.rss'
        ]
    
    def fetch_news(self, keyword: str, max_articles: int = 50) -> List[Dict]:
        """
        Fetch news articles related to cryptocurrency.
        
        Args:
            keyword: Crypto keyword to search
            max_articles: Maximum articles to fetch
            
        Returns:
            List of news articles
        """
        articles = []
        
        for feed_url in self.rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:max_articles]:
                    if keyword.lower() in entry.get('title', '').lower() or \
                       keyword.lower() in entry.get('summary', '').lower():
                        articles.append({
                            'title': entry.get('title', ''),
                            'summary': entry.get('summary', ''),
                            'link': entry.get('link', ''),
                            'published': entry.get('published', ''),
                            'source': feed_url
                        })
                
                logger.info(f"Fetched articles from {feed_url}")
                
            except Exception as e:
                logger.warning(f"Error fetching feed {feed_url}: {e}")
        
        return articles[:max_articles]
    
    def analyze_sentiment(self, texts: List[str]) -> float:
        """
        Analyze sentiment of news texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Average sentiment score (-1 to 1)
        """
        if not texts:
            return 0.0
        
        sentiments = []
        for text in texts:
            try:
                blob = TextBlob(text)
                sentiments.append(blob.sentiment.polarity)
            except Exception as e:
                logger.warning(f"Error analyzing news sentiment: {e}")
        
        return np.mean(sentiments) if sentiments else 0.0
    
    def get_crypto_news_sentiment(self, symbol: str, max_articles: int = 50) -> Dict:
        """
        Get aggregated news sentiment for a cryptocurrency.
        
        Args:
            symbol: Crypto symbol
            max_articles: Maximum articles to analyze
            
        Returns:
            News sentiment metrics
        """
        keyword = symbol.replace('-USD', '')
        articles = self.fetch_news(keyword, max_articles)
        
        if not articles:
            return {
                'symbol': symbol,
                'news_sentiment': 0.0,
                'article_count': 0,
                'timestamp': datetime.now()
            }
        
        # Analyze titles and summaries
        texts = [
            a['title'] + ' ' + a['summary']
            for a in articles
        ]
        
        sentiment_score = self.analyze_sentiment(texts)
        
        return {
            'symbol': symbol,
            'news_sentiment': sentiment_score,
            'article_count': len(articles),
            'recent_articles': articles[:3],
            'timestamp': datetime.now()
        }


class SentimentAggregator:
    """Aggregates sentiment from multiple sources."""
    
    def __init__(self, x_api_key: str, x_api_secret: str, x_bearer_token: str):
        """
        Initialize sentiment aggregator.
        
        Args:
            x_api_key: X API key
            x_api_secret: X API secret
            x_bearer_token: X bearer token
        """
        self.x_analyzer = XComSentimentAnalyzer(x_api_key, x_api_secret, x_bearer_token)
        self.news_analyzer = NewsSentimentAnalyzer()
    
    def get_combined_sentiment(
        self,
        symbol: str,
        x_weight: float = 0.3,
        news_weight: float = 0.2
    ) -> Dict:
        """
        Get combined sentiment score from all sources.
        
        Args:
            symbol: Crypto symbol
            x_weight: Weight for X.com sentiment
            news_weight: Weight for news sentiment
            
        Returns:
            Combined sentiment metrics
        """
        x_sentiment_data = self.x_analyzer.get_crypto_sentiment(symbol)
        news_sentiment_data = self.news_analyzer.get_crypto_news_sentiment(symbol)
        
        # Normalize weights
        weight_sum = x_weight + news_weight
        x_weight /= weight_sum
        news_weight /= weight_sum
        
        # Calculate combined score
        combined_score = (
            x_sentiment_data.get('sentiment_score', 0) * x_weight +
            news_sentiment_data.get('news_sentiment', 0) * news_weight
        )
        
        return {
            'symbol': symbol,
            'combined_sentiment': combined_score,
            'x_sentiment': x_sentiment_data.get('sentiment_score', 0),
            'x_mentions': x_sentiment_data.get('mention_count', 0),
            'news_sentiment': news_sentiment_data.get('news_sentiment', 0),
            'news_articles': news_sentiment_data.get('article_count', 0),
            'confidence': (abs(x_sentiment_data.get('sentiment_score', 0)) +
                          abs(news_sentiment_data.get('news_sentiment', 0))) / 2,
            'timestamp': datetime.now()
        }
