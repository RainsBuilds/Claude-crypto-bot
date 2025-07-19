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
from flask import Flask, render_template, jsonify, request
import threading
import queue

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
            'sandbox': False,  # Kraken doesn't support sandbox mode
        })
        
        # Trading settings
        self.portfolio_value = 1000  # Starting portfolio value
        self.max_position_size = 0.1  # Max 10% of portfolio per trade
        self.trading_pairs = ['XRP/USD', 'BTC/USD']
        self.paper_trading = True  # Set to False for real trading
        self.trading_active = True  # Can be toggled via web interface
        self.scan_interval = 300  # 5 minutes in seconds
        
        # Dashboard data
        self.last_decision = {}
        self.recent_decisions = []
        self.last_update = None
        
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
        
        # Create recent decisions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                action TEXT,
                symbol TEXT,
                amount REAL,
                reasoning TEXT,
                confidence INTEGER,
                executed BOOLEAN
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
    
    def get_portfolio_status(self) -> Dict:
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
        
        # Paper trading mode - just log the trade
        if self.paper_trading:
            price = current_prices.get(decision['symbol'], {}).get('price', 'N/A')
            logger.info(f"📝 PAPER TRADE: {decision['action']} ${decision['amount']} of {decision['symbol']} at ${price}")
            logger.info(f"💭 Reasoning: {decision['reasoning']}")
            return
            
        # Real trading mode
        try:
            if decision['action'] == 'BUY':
                # Execute buy order
                order = self.exchange.create_market_buy_order(
                    symbol, 
                    decision['amount']
                )
                logger.info(f"🚀 REAL BUY order executed: {order}")
                
            elif decision['action'] == 'SELL':
                # Execute sell order
                order = self.exchange.create_market_sell_order(
                    symbol,
                    decision['amount']
                )
                logger.info(f"🚀 REAL SELL order executed: {order}")
                
            # Record trade in database
            self.record_trade(decision, current_prices)
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
    
    def record_decision(self, decision: Dict, executed: bool = False):
        """Record Claude's decision in database"""
        conn = sqlite3.connect('trading_bot.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO decisions (timestamp, action, symbol, amount, reasoning, confidence, executed)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            decision.get('action', 'HOLD'),
            decision.get('symbol', ''),
            decision.get('amount', 0),
            decision.get('reasoning', ''),
            decision.get('confidence', 0),
            executed
        ))
        
        conn.commit()
        conn.close()
        
        # Update in-memory data for dashboard
        decision_with_time = decision.copy()
        decision_with_time['timestamp'] = datetime.now().strftime('%H:%M:%S')
        decision_with_time['executed'] = executed
        
        self.last_decision = decision_with_time
        self.recent_decisions.insert(0, decision_with_time)
        if len(self.recent_decisions) > 10:
            self.recent_decisions.pop()
    
    def get_recent_decisions(self, limit: int = 10) -> List[Dict]:
        """Get recent decisions from database"""
        conn = sqlite3.connect('trading_bot.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT timestamp, action, symbol, amount, reasoning, confidence, executed
            FROM decisions 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        decisions = []
        for row in cursor.fetchall():
            decisions.append({
                'timestamp': row[0],
                'action': row[1],
                'symbol': row[2],
                'amount': row[3],
                'reasoning': row[4],
                'confidence': row[5],
                'executed': bool(row[6])
            })
        
        conn.close()
        return decisions
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
        if not self.trading_active:
            logger.info("Trading is paused")
            return
            
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
        executed = False
        if decision.get('confidence', 0) >= 7:
            self.execute_trade(decision, price_data)
            executed = True
        else:
            logger.info(f"Confidence too low ({decision.get('confidence')}) - no trade executed")
        
        # Record decision
        self.record_decision(decision, executed)
        self.last_update = datetime.now()
    
    def run_bot(self):
        """Main bot loop"""
        logger.info("Starting Claude Crypto Trading Bot...")
        
        while True:
            try:
                self.run_trading_cycle()
                
                # Wait based on scan interval
                logger.info(f"Waiting {self.scan_interval // 60} minutes for next cycle...")
                time.sleep(self.scan_interval)
                
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(300)  # Wait 5 minutes on error

# Flask Web Dashboard
app = Flask(__name__, template_folder='templates')
bot = None

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/status')
def api_status():
    """Get current bot status"""
    try:
        if not bot:
            return jsonify({'error': 'Bot not initialized yet'}), 500
        
        portfolio = bot.get_portfolio_status()
        price_data = bot.get_price_data()
        
        # Calculate portfolio value with current prices
        total_value = portfolio.get('cash', 0)
        for symbol, position in portfolio.get('positions', {}).items():
            if symbol in price_data:
                total_value += position['amount'] * price_data[symbol]['price']
        
        return jsonify({
            'trading_active': bot.trading_active,
            'paper_trading': bot.paper_trading,
            'portfolio': {
                'cash': portfolio.get('cash', 0),
                'positions': portfolio.get('positions', {}),
                'total_value': total_value
            },
            'prices': price_data,
            'last_decision': getattr(bot, 'last_decision', {}),
            'recent_decisions': getattr(bot, 'recent_decisions', [])[:5],
            'last_update': bot.last_update.isoformat() if getattr(bot, 'last_update', None) else None,
            'scan_interval': bot.scan_interval
        })
    except Exception as e:
        logger.error(f"API status error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/toggle_trading', methods=['POST'])
def toggle_trading():
    """Toggle trading on/off"""
    try:
        if not bot:
            return jsonify({'error': 'Bot not initialized'}), 500
        
        bot.trading_active = not bot.trading_active
        status = "activated" if bot.trading_active else "paused"
        logger.info(f"Trading {status} via web interface")
        
        return jsonify({
            'trading_active': bot.trading_active,
            'message': f'Trading {status}'
        })
    except Exception as e:
        logger.error(f"Toggle trading error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/toggle_paper_mode', methods=['POST'])
def toggle_paper_mode():
    """Toggle paper trading mode"""
    try:
        if not bot:
            return jsonify({'error': 'Bot not initialized'}), 500
        
        bot.paper_trading = not bot.paper_trading
        mode = "paper" if bot.paper_trading else "live"
        logger.info(f"Switched to {mode} trading via web interface")
        
        return jsonify({
            'paper_trading': bot.paper_trading,
            'message': f'Switched to {mode} trading'
        })
    except Exception as e:
        logger.error(f"Toggle paper mode error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/set_interval', methods=['POST'])
def set_scan_interval():
    """Set scan interval"""
    try:
        if not bot:
            return jsonify({'error': 'Bot not initialized'}), 500
        
        data = request.get_json()
        interval_minutes = data.get('interval', 5)
        bot.scan_interval = interval_minutes * 60
        
        logger.info(f"Scan interval set to {interval_minutes} minutes via web interface")
        
        return jsonify({
            'scan_interval': bot.scan_interval,
            'message': f'Scan interval set to {interval_minutes} minutes'
        })
    except Exception as e:
        logger.error(f"Set interval error: {e}")
        return jsonify({'error': str(e)}), 500

def run_flask_app():
    """Run Flask app in a separate thread"""
    logger.info("Starting Flask dashboard on port 8080...")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=False, threaded=True)

def run_trading_bot():
    """Run trading bot in main thread"""
    global bot
    bot = CryptoTradingBot()
    bot.run_bot()

if __name__ == "__main__":
    # Start Flask app in background thread
    flask_thread = threading.Thread(target=run_flask_app, daemon=True)
    flask_thread.start()
    
    # Run trading bot in main thread
    run_trading_bot()
