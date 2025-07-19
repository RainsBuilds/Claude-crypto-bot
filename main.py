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
import pandas as pd
import numpy as np

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
            'sandbox': False,  # Set to True for testing
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
    
    def get_technical_analysis(self, symbol: str) -> Dict:
        """Get technical indicators for a symbol"""
        try:
            # Get 1-hour candles for the last 100 periods
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            if len(df) < 20:
                return {}
            
            # Calculate technical indicators
            close_prices = df['close']
            volumes = df['volume']
            
            # EMA (12 and 26 periods)
            ema_12 = close_prices.ewm(span=12).mean().iloc[-1]
            ema_26 = close_prices.ewm(span=26).mean().iloc[-1]
            ema_signal = "BULLISH" if ema_12 > ema_26 else "BEARISH"
            
            # RSI (14 periods)
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            if current_rsi > 70:
                rsi_signal = "OVERBOUGHT"
            elif current_rsi < 30:
                rsi_signal = "OVERSOLD"
            else:
                rsi_signal = "NEUTRAL"
            
            # Volume analysis (compare current to 20-period average)
            avg_volume = volumes.rolling(window=20).mean().iloc[-1]
            current_volume = volumes.iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            if volume_ratio > 2:
                volume_signal = "HIGH"
            elif volume_ratio > 1.5:
                volume_signal = "ELEVATED"
            else:
                volume_signal = "NORMAL"
            
            # Price change
            current_price = close_prices.iloc[-1]
            prev_price = close_prices.iloc[-2]
            price_change = ((current_price - prev_price) / prev_price) * 100
            
            return {
                'ema_12': round(ema_12, 2),
                'ema_26': round(ema_26, 2),
                'ema_signal': ema_signal,
                'rsi': round(current_rsi, 2),
                'rsi_signal': rsi_signal,
                'volume_ratio': round(volume_ratio, 2),
                'volume_signal': volume_signal,
                'price_change_1h': round(price_change, 2),
                'current_price': round(current_price, 2)
            }
            
        except Exception as e:
            logger.error(f"Error calculating technical analysis for {symbol}: {e}")
            return {}
        """Get current portfolio status from Kraken"""
        try:
            # Get real account balance from Kraken
            balance = self.exchange.fetch_balance()
            
            portfolio = {
                'cash': balance.get('USD', {}).get('free', 0) or 0,
                'positions': {},
                'total_value': 0
            }
            
            # Add crypto positions
            for symbol in ['XRP', 'BTC']:
                if balance.get(symbol, {}).get('total', 0) > 0:
                    portfolio['positions'][symbol] = {
                        'amount': balance[symbol]['total'],
                        'free': balance[symbol]['free'],
                        'used': balance[symbol]['used']
                    }
            
            # Calculate total portfolio value (simplified)
            portfolio['total_value'] = portfolio['cash'] or 0
            
            logger.info(f"Real portfolio: ${portfolio['cash']:.2f} cash, {len(portfolio['positions'])} positions")
            return portfolio
            
        except Exception as e:
            logger.error(f"Error fetching real balance: {e}")
            # Fallback to fake portfolio for testing
            return {
                'cash': 1000,
                'positions': {},
                'total_value': 1000
            }
    
    def analyze_with_claude(self, news_data: List[Dict], price_data: Dict, portfolio: Dict, technical_data: Dict) -> Dict:
        """Send data to Claude for analysis and trading decision"""
        
        # Prepare news summary
        news_summary = "\n".join([
            f"- {article['title']} ({article['source']})"
            for article in news_data[:5]
        ])
        
        # Prepare technical analysis summary
        tech_summary = ""
        for symbol in ['XRP', 'BTC']:
            if symbol in technical_data:
                tech = technical_data[symbol]
                tech_summary += f"\n{symbol} Technical Analysis:\n"
                tech_summary += f"  EMA Signal: {tech.get('ema_signal', 'N/A')} (12: {tech.get('ema_12', 'N/A')}, 26: {tech.get('ema_26', 'N/A')})\n"
                tech_summary += f"  RSI: {tech.get('rsi', 'N/A')} ({tech.get('rsi_signal', 'N/A')})\n"
                tech_summary += f"  Volume: {tech.get('volume_signal', 'N/A')} ({tech.get('volume_ratio', 'N/A')}x average)\n"
                tech_summary += f"  1h Price Change: {tech.get('price_change_1h', 'N/A')}%\n"
        
        # Create prompt for Claude
        prompt = f"""
        You are an expert cryptocurrency trading advisor. Analyze the market data and provide ONE specific trading recommendation.

        CURRENT MARKET DATA:
        XRP: ${price_data.get('XRP', {}).get('price', 'N/A')} (24h: {price_data.get('XRP', {}).get('change_24h', 'N/A')}%)
        BTC: ${price_data.get('BTC', {}).get('price', 'N/A')} (24h: {price_data.get('BTC', {}).get('change_24h', 'N/A')}%)

        TECHNICAL INDICATORS:
        {tech_summary}

        RECENT NEWS HEADLINES:
        {news_summary}

        PORTFOLIO STATUS:
        Available USD: ${portfolio.get('cash', 0):.2f}
        Total Portfolio Value: ${portfolio.get('total_value', 0):.2f}
        Current Positions: {portfolio.get('positions', {})}

        TRADING RULES:
        - Maximum position: $100 per trade
        - Only trade with confidence 7+ out of 10
        - Consider BOTH technical signals AND news sentiment
        - EMA crossovers and RSI levels are important
        - High volume can confirm breakouts
        - Provide specific reasoning

        REQUIRED FORMAT:
        ACTION: [BUY/SELL/HOLD]
        SYMBOL: [XRP or BTC]  
        AMOUNT: [Dollar amount like $50]
        REASONING: [2-3 sentences combining technical and fundamental analysis]
        CONFIDENCE: [Number 1-10]

        Example:
        ACTION: BUY
        SYMBOL: XRP
        AMOUNT: $75
        REASONING: EMA 12 crossed above EMA 26 showing bullish momentum, RSI at 45 indicates room to run higher. Fed master account news provides fundamental catalyst while technical setup confirms entry timing.
        CONFIDENCE: 8
        """
        
        try:
            response = self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
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
        
        # Convert symbol format for Kraken
        if symbol == 'BTC':
            symbol = 'BTC/USD'
        elif symbol == 'XRP':
            symbol = 'XRP/USD'
            
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
        
        # Get technical analysis for both symbols
        technical_data = {}
        for symbol in ['XRP/USD', 'BTC/USD']:
            technical_data[symbol.split('/')[0]] = self.get_technical_analysis(symbol)
        
        if not price_data:
            logger.error("No price data available - skipping cycle")
            return
            
        # Get Claude's analysis with technical data
        decision = self.analyze_with_claude(news_data, price_data, portfolio, technical_data)
        
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
