import os
import time
import requests
import json
import sqlite3
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import anthropic
import ccxt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CryptoTradingBot:
    def __init__(self):
        # API Keys from environment variables
        self.claude_api_key = os.getenv('CLAUDE_API_KEY')
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.kraken_api_key = os.getenv('KRAKEN_API_KEY')
        self.kraken_secret = os.getenv('KRAKEN_SECRET')
        
        # Initialize APIs
        self.claude_client = anthropic.Anthropic(api_key=self.claude_api_key)
        self.exchange = ccxt.kraken({
            'apiKey': self.kraken_api_key,
            'secret': self.kraken_secret,
            'sandbox': True,  # Set to True for testing
        })
        
        # Trading settings
        self.portfolio_value = 1000  # Starting portfolio value
        self.max_position_size = 0.1  # Max 10% of portfolio per trade
        self.trading_pairs = ['XRP/USD', 'BTC/USD']
        
        # Initialize database
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for tracking trades and portfolio"""
        conn = sqlite3.connect('trading_bot.db')
        cursor = conn.cursor()
        
        # Create trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                symbol TEXT,
                side TEXT,
                amount REAL,
                price REAL,
                reasoning TEXT,
                profit_loss REAL
            )
        ''')
        
        # Create portfolio table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio (
                symbol TEXT PRIMARY KEY,
                amount REAL,
                avg_price REAL,
                last_updated TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def get_news_data(self) -> List[Dict]:
        """Fetch crypto and XRP/BTC news"""
        news_data = []
        
        # NewsAPI - Crypto news
        try:
            url = f"https://newsapi.org/v2/everything"
            params = {
                'q': 'XRP OR Ripple OR Bitcoin OR cryptocurrency OR "Federal Reserve" OR "banking license"',
                'sortBy': 'publishedAt',
                'pageSize': 20,
                'language': 'en',
                'apiKey': self.news_api_key
            }
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                articles = response.json().get('articles', [])
                for article in articles[:10]:  # Top 10 articles
                    news_data.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'source': article.get('source', {}).get('name', ''),
                        'published': article.get('publishedAt', ''),
                        'url': article.get('url', '')
                    })
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            
        return news_data
    
    def get_price_data(self) -> Dict:
        """Get current prices for XRP and BTC"""
        try:
            ticker_xrp = self.exchange.fetch_ticker('XRP/USD')
            ticker_btc = self.exchange.fetch_ticker('BTC/USD')
            
            return {
                'XRP': {
                    'price': ticker_xrp['last'],
                    'change_24h': ticker_xrp['percentage'],
                    'volume': ticker_xrp['quoteVolume']
                },
                'BTC': {
                    'price': ticker_btc['last'],
                    'change_24h': ticker_btc['percentage'],
                    'volume': ticker_btc['quoteVolume']
                }
            }
        except Exception as e:
            logger.error(f"Error fetching prices: {e}")
            return {}
    
    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status"""
        conn = sqlite3.connect('trading_bot.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT symbol, amount, avg_price FROM portfolio')
        positions = cursor.fetchall()
        
        portfolio = {
            'cash': self.portfolio_value,
            'positions': {},
            'total_value': self.portfolio_value
        }
        
        for symbol, amount, avg_price in positions:
            portfolio['positions'][symbol] = {
                'amount': amount,
                'avg_price': avg_price
            }
            
        conn.close()
        return portfolio
    
    def analyze_with_claude(self, news_data: List[Dict], price_data: Dict, portfolio: Dict) -> Dict:
        """Send data to Claude for analysis and trading decision"""
        
        # Prepare news summary
        news_summary = "\n".join([
            f"- {article['title']} ({article['source']})"
            for article in news_data[:5]
        ])
        
        # Create prompt for Claude
        prompt = f"""
        You are a cryptocurrency trading advisor. Analyze the following data and provide a specific trading recommendation.

        CURRENT MARKET DATA:
        XRP Price: ${price_data.get('XRP', {}).get('price', 'N/A')}
        XRP 24h Change: {price_data.get('XRP', {}).get('change_24h', 'N/A')}%
        BTC Price: ${price_data.get('BTC', {}).get('price', 'N/A')}
        BTC 24h Change: {price_data.get('BTC', {}).get('change_24h', 'N/A')}%

        RECENT NEWS:
        {news_summary}

        CURRENT PORTFOLIO:
        Cash: ${portfolio.get('cash', 0)}
        Positions: {portfolio.get('positions', {})}

        TRADING RULES:
        - Maximum position size: 10% of portfolio value
        - Focus on XRP regulatory catalysts and BTC macro trends
        - Only trade when there's a clear edge
        - Always provide specific reasoning

        Provide your recommendation in this exact format:
        ACTION: [BUY/SELL/HOLD]
        SYMBOL: [XRP or BTC]
        AMOUNT: [$X or X coins]
        REASONING: [Your detailed analysis]
        CONFIDENCE: [1-10]
        """
        
        try:
            response = self.claude_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return self.parse_claude_response(response.content[0].text)
            
        except Exception as e:
            logger.error(f"Error calling Claude API: {e}")
            return {'action': 'HOLD', 'reasoning': 'API error'}
    
    def parse_claude_response(self, response_text: str) -> Dict:
        """Parse Claude's response into structured data"""
        decision = {
            'action': 'HOLD',
            'symbol': None,
            'amount': 0,
            'reasoning': 'No clear signal',
            'confidence': 5
        }
        
        lines = response_text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('ACTION:'):
                decision['action'] = line.split(':', 1)[1].strip()
            elif line.startswith('SYMBOL:'):
                decision['symbol'] = line.split(':', 1)[1].strip()
            elif line.startswith('AMOUNT:'):
                amount_str = line.split(':', 1)[1].strip()
                # Extract number from amount string
                try:
                    amount = float(''.join(filter(str.isdigit, amount_str.replace('.', 'X').replace('X', '.', 1))))
                    decision['amount'] = amount
                except:
                    decision['amount'] = 0
            elif line.startswith('REASONING:'):
                decision['reasoning'] = line.split(':', 1)[1].strip()
            elif line.startswith('CONFIDENCE:'):
                try:
                    decision['confidence'] = int(line.split(':', 1)[1].strip())
                except:
                    decision['confidence'] = 5
                    
        return decision
    
    def execute_trade(self, decision: Dict, current_prices: Dict):
        """Execute the trading decision"""
        if decision['action'] == 'HOLD':
            logger.info("Claude recommends HOLD - no trade executed")
            return
            
        symbol = decision['symbol']
        if symbol not in self.trading_pairs:
            logger.error(f"Invalid symbol: {symbol}")
            return
            
        try:
            if decision['action'] == 'BUY':
                # Execute buy order
                order = self.exchange.create_market_buy_order(
                    symbol, 
                    decision['amount']
                )
                logger.info(f"BUY order executed: {order}")
                
            elif decision['action'] == 'SELL':
                # Execute sell order
                order = self.exchange.create_market_sell_order(
                    symbol,
                    decision['amount']
                )
                logger.info(f"SELL order executed: {order}")
                
            # Record trade in database
            self.record_trade(decision, current_prices)
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
    
    def record_trade(self, decision: Dict, current_prices: Dict):
        """Record trade in database"""
        conn = sqlite3.connect('trading_bot.db')
        cursor = conn.cursor()
        
        price = current_prices.get(decision['symbol'].split('/')[0], {}).get('price', 0)
        
        cursor.execute('''
            INSERT INTO trades (timestamp, symbol, side, amount, price, reasoning)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            decision['symbol'],
            decision['action'],
            decision['amount'],
            price,
            decision['reasoning']
        ))
        
        conn.commit()
        conn.close()
    
    def run_trading_cycle(self):
        """Run one complete trading cycle"""
        logger.info("Starting trading cycle...")
        
        # Gather data
        news_data = self.get_news_data()
        price_data = self.get_price_data()
        portfolio = self.get_portfolio_status()
        
        if not price_data:
            logger.error("No price data available - skipping cycle")
            return
            
        # Get Claude's analysis
        decision = self.analyze_with_claude(news_data, price_data, portfolio)
        
        logger.info(f"Claude decision: {decision}")
        
        # Execute trade if confidence is high enough
        if decision.get('confidence', 0) >= 7:
            self.execute_trade(decision, price_data)
        else:
            logger.info(f"Confidence too low ({decision.get('confidence')}) - no trade executed")
    
    def run_bot(self):
        """Main bot loop"""
        logger.info("Starting Claude Crypto Trading Bot...")
        
        while True:
            try:
                self.run_trading_cycle()
                
                # Wait 1 hour before next cycle
                logger.info("Waiting 1 hour for next cycle...")
                time.sleep(3600)
                
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(300)  # Wait 5 minutes on error

if __name__ == "__main__":
    bot = CryptoTradingBot()
    bot.run_bot()
