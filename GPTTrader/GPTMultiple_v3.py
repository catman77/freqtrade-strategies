import requests
from openai import OpenAI
import tiktoken

from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.freqai_interface import IFreqaiModel
from datetime import datetime, timedelta, timezone
from bs4 import BeautifulSoup
import urllib
import pytz
import re
import praw
from requests import Session
from pandas import DataFrame
from typing import Any, Tuple, Dict
import logging
import time
import numpy as np
import json
import statistics
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)


"""
The following freqaimodel is released to sponsors of the non-profit FreqAI open-source project.
If you find the FreqAI project useful, please consider supporting it by becoming a sponsor.
We use sponsor money to help stimulate new features and to pay for running these public
experiments, with a an objective of helping the community make smarter choices in their
ML journey.

This strategy is experimental (as with all strategies released to sponsors). Do *not* expect
returns. The goal is to demonstrate gratitude to people who support the project and to
help them find a good starting point for their own creativity.

If you have questions, please direct them to our discord: https://discord.gg/xE4RMg4QYw

https://github.com/sponsors/robcaulk
"""


class GPTMultiple_v3(IFreqaiModel):
    """
    Base model for letting an LLM take full control of your bot...
    """

    def __init__(self, **kwargs):
        super().__init__(config=kwargs["config"])

        """
        If you prefer to use a locally hosted LLM e.g. (oobabooga/text-generation-web-ui) ensure you're running the latest version of it, add to your config.
        "GPTTrader": {
            "customgpt_use": true, // False for ChatGPT
            "customgpt_base_url": "http://127.0.0.1:5000/v1",
            "customgpt_api_key": "sk-111111111111111111111111111111111111111111111111",
            "customgpt_model": "TheBloke_Mixtral_7Bx2_MoE-GPTQ_gptq-4bit-32g-actorder_True", // Change to the model your running. To list models: curl -H "Authorization: Bearer sk-111111111111111111111111111111111111111111111111" http://127.0.0.1:5000/v1/models

            https://github.com/oobabooga/text-generation-webui/wiki/12-%E2%80%90-OpenAI-API#third-party-application-setup
            e.g. Linux (run once):
            export OPENAI_API_KEY=sk-111111111111111111111111111111111111111111111111
            export OPENAI_API_BASE=http://127.0.0.1:5000/v1

            ./start_linux.sh --auto-devices --model TheBloke_Mixtral_7Bx2_MoE-GPTQ_gptq-4bit-32g-actorder_True --model-menu --extensions openai --auto-launch --api --api-key sk-111111111111111111111111111111111111111111111111
        """
        self.customgpt_use = self.freqai_info['GPTTrader'].get(
            'customgpt_use', True)

        self.customgpt_base_url = self.freqai_info['GPTTrader'].get(
            'customgpt_base_url', "")

        self.customgpt_api_key = self.freqai_info['GPTTrader'].get(
            'customgpt_api_key', "")

        self.customgpt_model = self.freqai_info['GPTTrader'].get(
            'customgpt_model', "")

        """
        If using OpenAI's ChatGPT. 
        "GPTTrader": {
            "customgpt_use": false,
            "openai_key": "sk-...."
            "gpt_model": "gpt-3.5-turbo", // or gpt-4 if you want to use gpt4
        """

        self.openai_key = self.freqai_info['GPTTrader'].get("openai_key", None)
        if self.openai_key is None and self.customgpt_use == False:
            raise ValueError("openai_key is not set in freqtrade config.json")
        else:
            openai_api_key = self.openai_key

        self.gpt_model = self.freqai_info["GPTTrader"].get(
            "gpt_model", "gpt-3.5-turbo")
        self.news_hours = self.freqai_info["GPTTrader"].get("news_hours", 6)

        self.redditapi_client_id = self.freqai_info['GPTTrader'].get(
            'redditapi_client_id', "")

        self.redditapi_client_secret = self.freqai_info['GPTTrader'].get(
            'redditapi_client_secret', "")

        self.asknews_api_key = self.freqai_info['GPTTrader'].get(
            'asknews_api_key', "")

        self.asknews_tokens = self.freqai_info['GPTTrader'].get(
            'asknews_tokens', [])

        # Options for setting coin_dict tradingmode for spot or futures
        self.tradingmode = self.config.get("trading_mode", "spot")
        self.coin_dict = set_coin_dict(self, self.tradingmode)

        # Get the list of pairs for training
        self.pairs = list(self.coin_dict.keys())

        self.news_cache = {}  # Dictionary to cache articles for each news provider

        self.news_providers = {
            "google": "https://news.google.com/rss/search?q={}",
            "bing": "https://www.bing.com/news/search?q={}&format=rss&qft=interval%3d4+sortbydate%3d1&form=PTFTNR",
            "coindesk": "https://www.coindesk.com/feed",
            "cointelegraph": "https://cointelegraph.com/feed",
            "cryptoslate": "https://cryptoslate.com/feed/",
            "newsbtc": "https://www.newsbtc.com/feed/",
            "cryptobriefing": "https://cryptobriefing.com/feed/",
            "bitcoinmagazine": "https://bitcoinmagazine.com/feed",
            "decrypt": "https://decrypt.co/feed",
            "coincodecap": "https://coincodecap.com/category/news/feed/gn",
            "bitcoin": "https://news.bitcoin.com/feed/",
            "cryptonewsz": "https://www.cryptonewsz.com/feed/",
            "coinsutra": "https://coinsutra.com/blog/feed/",
            "blockonomi": "https://blockonomi.com/feed/",
            "coinspeaker": "https://feeds.feedburner.com/coinspeaker/",
            "yahoofinance": "https://finance.yahoo.com/rss/crypto/",
            "bloombergcrypto": "https://news.google.com/rss/search?q=cryptocurrency+site:bloomberg.com",
            "chainalysis": "https://blog.chainalysis.com/feed",
            # "cryptodash": "https://cryptodash.com/rss", 522 error
            "cryptonews": "https://cryptonews.com/news/feed/",
            "marketbeat": "https://www.marketbeat.com/cryptocurrencies/news/",
            "fxstreet": "https://www.fxstreet.com/cryptocurrencies/news",
            "spotonchain": "https://platform.spotonchain.ai/",
            "santiment": "https://sanr.app/signals"
        }

        # Copy coin_dict to a temporary dictionary for cache
        self.cache_coin_dict = dict(self.coin_dict)

    def clear_cache(self):
        self.news_cache = {}
        self.cache_coin_dict = dict(self.coin_dict)

    def train(
        self, unfiltered_df: DataFrame, pair: str, dk: FreqaiDataKitchen, **kwargs
    ) -> Any:

        search_query = self.coin_dict[pair]
        token = pair.split("/")[0]
        token_name = extract_token_name(token, search_query)

        """Build a custom prompt to determine news sentiment and provide an action based on the current coin position."""

        # Using LocalGPT
        if self.customgpt_use:
            # Attributes for Llama2 model
            self.prompt = ("""
### Instruction:

You are a helpful, respectful and honest crypto trading assistant who is risk tolerant and analyzes news or posts sentiment, provid trading recommendations.
For each news headline or post provided, you clearly specify its sentiment using ONLY `NEGATIVE`, `NEUTRAL`, or `POSITIVE`. Then provide an overall sentiment based on the collective assessments.
Provide one trading recommendation taking into account the overall sentiment, current position, stop-loss, target and trading duration.
The ONLY Options for a trading recommendation are:
 LONG_ENTER
 LONG_EXIT
 SHORT_ENTER
 SHORT_EXIT
 NEUTRAL

LONG_ENTER = Enter a long position when the sentiment is primarily positive and I am not in an existing position.
LONG_EXIT = Exit a long position when the sentiment turns negative and I'm in a long position.
SHORT_ENTER = Enter a short position when the sentiment is primarily negative and I'm not in an existing position.
SHORT_EXIT = Exit a short position when the sentiment becomes positive and I'm in a short position.
NEUTRAL = Maintain a neutral position when the sentiment is mixed or unclear or there is insufficient information.
Conclude with one sentence max final summary opinion on the sentiment analysis result and your reasoning. Ensure you use the following format.
<Index>: <Article/Post/News> - <SENTIMENT>
Trading Recommendation: <ACTION>
Final Summary Opinion: <Reasoning>

Ensure the format is correct, and exactly as above. Check the Trading Recommendation: <ACTION> is accurate, aligned with current trading position and overall sentiment.
""")

            self.request = (
                f"### INPUT ###\nAnalyze sentiment from news or posts about {token_name} ({token})")

            self.buy_sell = ("")

            if "" in self.customgpt_model:
                self.summary = """<|end_of_turn|>
    
    ### Response:
    
    """
            else:
                self.summary = """
    
    ### Response:
    
    """

        # Using OpenAI
        else:
            self.prompt = (f"Analyze sentiment from news or posts about {token_name} ({token}). "
                           "Your response should include a clear trading recommendation under 'Recommended Action:'.")

            self.request = ("""For each news headline or posts, classify its sentiment as `NEGATIVE`, `NEUTRAL`, or `POSITIVE`. """
                            "Summarize your findings as: <Index>: <Content> - <SENTIMENT>.")

            self.buy_sell = ("""
            After assessing the news items, deduce the overall sentiment and provide a trading recommendation. 
            Options for the trading recommendation <ACTION> are:
            - `LONG_ENTER` for mainly POSITIVE sentiment without an existing position.
            - `LONG_EXIT` for mainly NEGATIVE sentiment when holding a long position.
            - `SHORT_ENTER` for mainly NEGATIVE sentiment without a current position.
            - `SHORT_EXIT` for mainly POSITIVE sentiment when holding a short position.
            - `NEUTRAL` for mixed or non-committal sentiments.
            Ensure you format as: 'Recommended Action: <ACTION>'. 
            """)

            self.summary = ("Wrap up with a succinct rationale (1 sentence) on the sentiment analysis result. "
                            "Present it as: 'Final Summary Opinion: <Reasoning>'.")

        side, profit, duration = self.get_state_info(pair)
        target_profit_percentage = int(
            self.freqai_info['GPTTrader'].get('target_profit', 0.03) * 100)
        stoploss_percentage = int(
            self.freqai_info['GPTTrader'].get('stoploss', 0.04) * 100)
        current_profit_percentage = profit * 100

        if side == 0:
            side = "short"
        if side == 1:
            side = "long"
        if side == 0.5:
            side = "not in a trade"

        if side == "not in a trade":
            self.position = (f"I am not currently in a trade."
                             f" The current date is {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}\n\n")
        else:
            gain_loss_text = (f"and I have achieved a profit of {current_profit_percentage}%, which is below my target profit of {target_profit_percentage}%" if current_profit_percentage >= 0
                              else f"and I am at a loss of -{abs(current_profit_percentage)}%.")
            self.position = (f"My current position is {side}, {gain_loss_text}. The trade has been open for {duration} candles (" +
                             str(int(round(self.freqai_info['live_retrain_hours'] * 60 / 1, 0))) +
                             f"min each), and the current date is {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}.\n\n"
                             f"My target duration for the trade is {self.freqai_info['GPTTrader'].get('target_duration_candles', 100)} candles, with each candle representing {int(round(self.freqai_info['live_retrain_hours'] * 60 / 1, 0))} minutes, and I have set a stop-loss at {stoploss_percentage}%.\n\n")

        self.desires = (f"My target profit is {target_profit_percentage}% "
                        f"and my target duration is {self.freqai_info['GPTTrader'].get('target_duration_candles', 100)} candles, " +
                        "each candle is " +
                        str(int(
                            round(self.freqai_info['live_retrain_hours'] * 60 / 1, 0))) + "mins "
                        f"and my stop-loss is {stoploss_percentage}%.\n\n")

        logger.info("++++ " + colorize_text("Pair: " + pair, 'green') + " ++++")

        fetched_reddit_posts, reddit_prompt, reddit_post_count = fetch_reddit_posts(
            tokens=[token, token_name], max_hours_old=self.news_hours, reddit_client_id=self.redditapi_client_id, reddit_client_secret=self.redditapi_client_secret)

        fetched_coinmarketcapcommunity_posts, coinmarketcapcommunity_prompt, coinmarketcapcommunity_post_count = fetch_coinmarketcap_gravity_posts(
            tokens=[token, token_name], max_hours_old=self.news_hours)

        articles, headlines, headline_count = fetch_articles(
            search_query, news_providers=self.news_providers, max_hours_old=self.news_hours,
            max_articles=20, tokens=[token, token_name], news_cache=self.news_cache)

        if self.asknews_api_key != "":
            try:
                asknews_sentiment, sentiment_headline = fetch_asknews_sentiment(
                    token_name, self.news_hours, self.asknews_api_key, self.asknews_tokens)
                if asknews_sentiment:
                    if len(articles) == 0:
                        headline_count = 1
                    else:
                        headline_count += 1
                    headlines += str(headline_count) + \
                        ": " + sentiment_headline
                    articles.extend(asknews_sentiment)
            except Exception as e:
                logger.warning(e)

        # Remove the selected pair from the cache coin_dict dictionary
        # logger.info("++++ " + colorize_text("cache_coin_dict length: " + str(len(self.cache_coin_dict)), 'yellow') + " " + str(self.cache_coin_dict) + " ++++")
        # Check if the pair exists in cache_coin_dict
        if pair in self.cache_coin_dict:
            # Remove the selected pair from the cache coin_dict dictionary
            del self.cache_coin_dict[pair]

            # Check if it's the last pair
            if not self.cache_coin_dict:
                # Clear the cache after processing the last pair
                self.clear_cache()
        else:
            # logger.info(f"Pair {pair} not found in cache_coin_dict")
            pass

        try:
            if "train_features" not in dk.data_dictionary:
                dk.data_dictionary["train_features"] = DataFrame(np.zeros(1))
                logger.warning(
                    "train_features data not found in data dictionary, assigning default value.")
        except KeyError as e:
            logger.error(
                f"An error occurred while accessing 'train_features': {e}")

        dk.data_dictionary["train_dates"] = DataFrame(np.zeros(1))
        for label in dk.label_list:
            dk.data["labels_mean"] = {}
            dk.data["labels_mean"][label] = 0
            dk.data["labels_std"] = {}
            dk.data["labels_std"][label] = 0

        if len(articles) == 0 and len(fetched_reddit_posts) == 0 and len(fetched_coinmarketcapcommunity_posts) == 0:
            logger.info(
                f"No articles, Reddit posts, or CoinMarketCap Community posts found for {search_query}")
            return {"sentiment_yes": 0, "sentiment_no": 0, "sentiment_unknown": 0,
                    "expert_long_enter": 0, "expert_long_exit": 0,
                    "expert_short_enter": 0, "expert_short_exit": 0,
                    "expert_neutral": 0, "expert_opinion": ""}

        content = ""
        if len(articles) != 0 and len(fetched_reddit_posts) == 0 and len(fetched_coinmarketcapcommunity_posts) == 0:
            content = self.analyze_sentiment(headlines + "\n")
        elif len(articles) == 0 and len(fetched_reddit_posts) != 0 and len(fetched_coinmarketcapcommunity_posts) == 0:
            content = self.analyze_sentiment(reddit_prompt + "\n")
        elif len(articles) == 0 and len(fetched_reddit_posts) == 0 and len(fetched_coinmarketcapcommunity_posts) != 0:
            content = self.analyze_sentiment(
                coinmarketcapcommunity_prompt + "\n")
        elif len(articles) != 0 and len(fetched_reddit_posts) != 0 and len(fetched_coinmarketcapcommunity_posts) == 0:
            content = self.analyze_sentiment(reddit_prompt + headlines + "\n")
        elif len(articles) != 0 and len(fetched_reddit_posts) == 0 and len(fetched_coinmarketcapcommunity_posts) != 0:
            content = self.analyze_sentiment(
                coinmarketcapcommunity_prompt + headlines + "\n")
        elif len(articles) == 0 and len(fetched_reddit_posts) != 0 and len(fetched_coinmarketcapcommunity_posts) != 0:
            content = self.analyze_sentiment(
                reddit_prompt + coinmarketcapcommunity_prompt + "\n")
        else:
            content = self.analyze_sentiment(
                reddit_prompt + headlines + coinmarketcapcommunity_prompt + "\n")

        model = evaluate_content(content, pair)
        return model

    def predict(
        self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs
    ):
        """
        Get the cached predictions and feed them back to the strategy
        """
        dk.data['extra_returns_per_train']['sentiment_yes'] = self.model["sentiment_yes"]
        dk.data['extra_returns_per_train']['sentiment_no'] = self.model["sentiment_no"]
        dk.data['extra_returns_per_train']['sentiment_unknown'] = self.model["sentiment_unknown"]
        dk.data['extra_returns_per_train']['expert_long_enter'] = self.model["expert_long_enter"]
        dk.data['extra_returns_per_train']['expert_long_exit'] = self.model["expert_long_exit"]
        dk.data['extra_returns_per_train']['expert_short_enter'] = self.model["expert_short_enter"]
        dk.data['extra_returns_per_train']['expert_short_exit'] = self.model["expert_short_exit"]
        dk.data['extra_returns_per_train']['expert_neutral'] = self.model["expert_neutral"]
        dk.data['extra_returns_per_train']['expert_opinion'] = self.model["expert_opinion"]

        # we will simply return zeros for the classical predictions
        zeros = len(unfiltered_df.index)
        return DataFrame(np.zeros(zeros), columns=["&-empty"]), np.ones(zeros)

    def analyze_sentiment(self, text) -> Dict[str, int]:

        # Using Locally hosted LLM with text-generation-web api
        if self.customgpt_use == True:
            logger.info("++++ " + colorize_text("Using Custom GPT...", 'cyan'))
            logger.info("++++ " + colorize_text("Analysing Sentiment:", 'green') + " ++++ " + str(self.prompt) + str(
                self.request) + str(text) + str(self.buy_sell) + str(self.position) + str(self.desires) + str(self.summary))

            input_prompt = str(self.prompt) + str(self.request) + str(text) + str(
                self.buy_sell) + str(self.position) + str(self.desires) + str(self.summary)
            formatted_prompt = input_prompt.replace(
                '\n', '\\n').replace('\t', '\\t')

            if self.customgpt_base_url == "":
                raise ValueError(
                    "using localgpt and missing base url in freqtrade config.json")
            if self.customgpt_api_key == "":
                raise ValueError(
                    "using localgpt and missing api key in freqtrade config.json")
            if self.customgpt_model == "":
                raise ValueError(
                    "using localgpt and missing model in freqtrade config.json")

            openai_base_url = self.customgpt_base_url
            openai_api_key = self.customgpt_api_key
            openai_model = self.customgpt_model

            client = OpenAI(
                base_url=openai_base_url,
                api_key=openai_api_key
            )

            response = client.chat.completions.create(
                model=openai_model,
                messages=[
                    {"role": "user", "content": formatted_prompt},
                ],
                max_tokens=512,
                temperature=0.1,
                top_p=1,
            )
            result = response.choices[0].message.content

            """
            endpoint_url = openai_base_url + "/chat/completions"
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openai_api_key}"
            }
            
            history = []
            history.append({"role": "user", "content": formatted_prompt})
            if "127.0.0.1" in endpoint_url or "localhost" in endpoint_url:
                data = {
                    "mode": "instruct",
                    "messages": history,
                    "max_tokens": 512,
                    "temperature": 0.7,
                    "top_p": 0.99,
                    "typical_p": 1,
                    "top_k": 80,
                    "repetition_penalty": 1.1
                }
            else:
                data = {
                    "messages": history,
                    "max_tokens": 512,
                    "temperature": 0.7,
                    "top_p": 0.99,
                    #"typical_p": 1,
                    #"top_k": 80,
                    #"repetition_penalty": 1.1
            }
            response = requests.post(endpoint_url, headers=headers, json=data, verify=False)
            result = response.json()['choices'][0]['message']['content']
            """
            content = result.replace('\\n', '\n').replace('\\t', '\t')

            logger.info("++++ " + colorize_text("Sentiment Response:",
                        'green') + " ++++ " + content)

        # Using OpenAI GPT
        else:
            logger.info("++++ " + colorize_text("Using OpenAI...", 'cyan'))
            token_count = num_tokens_from_string(
                self.prompt, "cl100k_base", self.gpt_model)
            # Count the tokens
            cost = 0
            if self.gpt_model == "gpt-3.5-turbo":
                cost = token_count/1000 * 0.0015
            elif self.gpt_model == "gpt-4":
                cost = token_count/1000 * 0.03
            formatted_cost = "{:.6f}".format(cost)
            logger.info(
                f"Input Token count: {token_count}, which will cost ${formatted_cost} USD")

            retries = 0
            while retries < 5:
                try:
                    client = OpenAI(
                        api_key=openai_api_key
                    )
                    response = client.chat.completions.create(
                        model=self.gpt_model,
                        messages=[
                            {"role": "system", "content": self.prompt},
                            {"role": "user", "content": self.request + text +
                                self.buy_sell + self.position + self.desires + self.summary},
                        ],
                        temperature=0,
                    )
                    break
                except openai.error.RateLimitError:
                    logger.warning(
                        "RateLimitError: That model is currently overloaded with other requests. Retrying in 10 seconds.")
                    time.sleep(10)
                    retries += 1

            if retries == 5:
                raise Exception(
                    "Failed to get response from OpenAI after 5 retries.")

            content = response['choices'][0]['message']['content']

            token_count = num_tokens_from_string(
                content, "cl100k_base", self.gpt_model)
            logger.info(
                f"Output Token count: {token_count}, which will cost ${token_count/1000 * 0.002} USD")

        return content

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        return None

    def get_state_info(self, pair: str) -> Tuple[float, float, int]:
        """
        State info during dry/live (not backtesting) which is fed back
        into the model.
        :param pair: str = COIN/STAKE to get the environment information for
        :return:
        :market_side: float = representing short, long, or neutral for
            pair
        :current_profit: float = unrealized profit of the current trade
        :trade_duration: int = the number of candles that the trade has
            been open for
        """
        open_trades = Trade.get_trades_proxy(is_open=True)
        market_side = 0.5
        current_profit: float = 0
        trade_duration = 0
        for trade in open_trades:
            if trade.pair == pair:
                if self.data_provider._exchange is None:  # type: ignore
                    logger.error('No exchange available.')
                    return 0, 0, 0
                else:
                    current_rate = self.data_provider._exchange.get_rate(  # type: ignore
                        pair, refresh=False, side="exit", is_short=trade.is_short)

                now = datetime.now(timezone.utc).timestamp()
                trade_duration = int(
                    (now - trade.open_date_utc.timestamp()) / self.base_tf_seconds)
                current_profit = trade.calc_profit_ratio(current_rate)

                if trade.is_short:
                    market_side = 0
                else:
                    market_side = 1

        return market_side, current_profit, int(trade_duration)


def extract_token_name(token, search_query):
    if token.lower() in search_query.lower():
        index = search_query.lower().rindex(token.lower())
        search_string = search_query[:index].strip()
        if search_string == "":
            return token
        else:
            return search_string
    else:
        return token


def colorize_text(text, text_color=None, bg_color=None, bold=False):
    color_dict = {
        'black': '0',
        'red': '1',
        'green': '2',
        'yellow': '3',
        'blue': '4',
        'magenta': '5',
        'cyan': '6',
        'white': '7'
    }

    color_text = text
    color_code = ''

    if bold:
        color_code += '\033[1m'

    if text_color and text_color.lower() in color_dict:
        color_code += f'\033[1;3{color_dict[text_color.lower()]}m'

    if bg_color and bg_color.lower() in color_dict:
        color_code += f'\033[1;4{color_dict[bg_color.lower()]}m'

    if color_code != '':
        # resets color and style to terminal default
        color_text = color_code + text + '\033[0m'

    return color_text


def evaluate_content(content, pair):
    content_lower = content.lower()  # Convert content to lowercase
    content_lower = content_lower.replace(
        '*', '')  # Remove asterisks if present
    content = content.replace('*', '')  # Remove asterisks if present

    # count number of occurrences of "POSITIVE", "NEGATIVE", "NEUTRAL"
    expert_model = {}
    expert_model["sentiment_yes"] = content_lower.count("positive")
    expert_model["sentiment_no"] = content_lower.count("negative")
    expert_model["sentiment_unknown"] = (content_lower.count(
        "neutral") - content_lower.count("recommended action: neutral"))

    # Check for trading recommendations based on the expected format
    keywords = ["long_enter", "long enter", "long_exit", "long exit",
                "short_enter", "short enter",  "short_exit", "short exit", "neutral"]

    expert_model["expert_long_enter"] = any(
        keyword in content_lower for keyword in ["recommended action: long_enter", "trading recommendation: long_enter", "recommended action: long enter", "trading recommendation: long enter"])
    expert_model["expert_long_exit"] = any(
        keyword in content_lower for keyword in ["recommended action: long_exit", "trading recommendation: long_exit", "recommended action: long exit", "trading recommendation: long exit"])
    expert_model["expert_short_enter"] = any(
        keyword in content_lower for keyword in ["recommended action: short_enter", "trading recommendation: short_enter", "recommended action: short enter", "trading recommendation: short enter"])
    expert_model["expert_short_exit"] = any(
        keyword in content_lower for keyword in ["recommended action: short_exit", "trading recommendation: short_exit", "recommended action: short exit", "trading recommendation: short exit"])
    expert_model["expert_neutral"] = any(
        keyword in content_lower for keyword in ["recommended action: neutral", "trading recommendation: neutral"])

    keyword_found = any(
        keyword in content_lower for keyword in keywords)
    expert_model["expert_opinion"] = None

    if keyword_found:
        # Look for the "Final Summary Opinion:" if a keyword is found
        final_summary_start = content_lower.find("final summary opinion:")

        if final_summary_start != -1:
            # Determine the correct case of the substring
            final_summary_prefix = content[final_summary_start:final_summary_start + len(
                "Final Summary Opinion:")].strip()
            if final_summary_prefix.lower() == "final summary opinion:":
                expert_model["expert_opinion"] = content[final_summary_start +
                                                         len("Final Summary Opinion:"):].strip()
            elif final_summary_prefix.upper() == "FINAL SUMMARY OPINION:":
                expert_model["expert_opinion"] = content[final_summary_start +
                                                         len("FINAL SUMMARY OPINION:"):].strip()
        else:
            for keyword in keywords:
                if any(keyword in content_lower for kw in ["recommended action: " + keyword, "trading recommendation: " + keyword]):
                    split_content = content.split(keyword)

                    # Check if the content after the keyword is empty
                    if split_content[-1].strip() == '':
                        # Take the content before the keyword
                        expert_model["expert_opinion"] = split_content[-2]
                    else:
                        # Take the content after the keyword as usual
                        expert_model["expert_opinion"] = split_content[-1]

                    # Check if the expert_opinion starts with a full stop and remove it
                    if expert_model["expert_opinion"].lstrip().startswith('.'):
                        expert_model["expert_opinion"] = expert_model["expert_opinion"].lstrip()[
                            1:]

                    # Remove all leading line breaks
                    expert_model["expert_opinion"] = expert_model["expert_opinion"].lstrip(
                        '\n')

    # Log the opinion
    logger.info(f"Expert Opinion on {pair}: {expert_model['expert_opinion']}")

    return expert_model


def num_tokens_from_string(string: str, encoding_name: str, gptmodel: str) -> int:
    """Returns the number of tokens in a text string."""

    encoding = tiktoken.encoding_for_model(gptmodel)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def replace_non_bmp_characters(input_string, replacement_char=' '):
    """
    Replace all non-BMP (Basic Multilingual Plane) characters in the input string
    with the specified replacement character.

    Args:
        input_string (str): The input string to process.
        replacement_char (str): The character to replace non-BMP characters with.

    Returns:
        str: The input string with non-BMP characters replaced.
    """
    return ''.join(char if ord(char) < 65536 else replacement_char for char in input_string)


def fetch_articles(search_query, news_providers, max_hours_old, max_articles, tokens, news_cache):
    articles = []
    prompt = "\nNews:\n"
    headline_number = 1  # Initialize the headline number

    for news_provider in news_providers:
        logger.info("+++++++ " + colorize_text("Checking News Provider:", 'magenta') + " " + news_provider + " " +
                    colorize_text("Search Tokens: " + tokens[0] + " | " + tokens[1], 'magenta') + " in title/description")

        if news_provider in ["google", "bing"]:
            # For Google and Bing, include the news provider name in the cache key
            cache_key = f"{news_provider}_{search_query}"
        else:
            # For other providers, use a single cache key for all tokens
            cache_key = news_provider

        # Check if the news provider data is already in the cache
        if cache_key in news_cache:
            provider_articles = news_cache[cache_key]
        else:
            # Fetch the articles and store them in the cache
            provider_articles = parse_articles_from_provider(
                news_providers, news_provider, search_query, max_hours_old, max_articles, tokens)
            # Ensure 'provider_headlines' is a list even when fetched from parsing
            news_cache[cache_key] = (provider_articles)

        # Filter the cached articles to include only those that match the tokens
        matching_articles = []
        for article in provider_articles:
            title = article["title"]
            description = article["description"]
            pub_date_str = article["pub_date"]
            # or any(re.search(r'\b' + re.escape(token.lower()) + r'\b', description.lower()) for token in tokens):
            if any(re.search(r'\b' + re.escape(token.lower()) + r'\b', title.lower()) for token in tokens):

                logger.info("++++++++ " + colorize_text("Found:",
                                                        'green') + " " + article["title"])
                # logger.info("++++++++ Found: " + article["description"])
                prompt += f"{headline_number}: Title: {article['title']}, published: {pub_date_str}\n"

                matching_articles.append(article)
                headline_number += 1

        articles.extend(matching_articles)

    return articles, prompt, headline_number


def parse_articles_from_provider(news_providers, news_provider, search_query, max_hours_old, max_articles, tokens):
    if news_provider == "google" or news_provider == "bing":
        url = news_providers[news_provider].format(search_query)
    elif news_provider in news_providers:
        url = news_providers[news_provider]
    else:
        valid_providers = "', '".join(news_providers.keys())
        raise ValueError(f"Invalid news provider. Use '{valid_providers}'.")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Edg/115.0.1901.183",
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        articles = parse_articles_from_response(
            response.text, news_provider, max_hours_old, max_articles, tokens)
        return articles
    else:
        logger.error(colorize_text("Error fetching articles from {}: Status code: {}".format(
            news_provider, response.status_code), 'red'))
        return []


def parse_articles_from_response(xml_content, news_provider, max_hours_old, max_articles, tokens):
    from xml.etree import ElementTree as ET

    articles = []
    if news_provider != "marketbeat" and news_provider != "fxstreet" and news_provider != "spotonchain" and news_provider != "santiment":
        root = ET.fromstring(xml_content)

        if news_provider == "google":
            namespace = None
        elif news_provider == "bing":
            namespace = {"news": "http://www.bing.com/schema/news"}
        else:
            namespace = None  # For other providers, namespace is not required

        # Use the 'GMT' time zone for conversion
        gmt_timezone = pytz.timezone('GMT')

        for item in root.findall(".//item"):
            title = item.find("title").text
            link = item.find("link").text

            description_element = item.find("description")
            if description_element is not None:
                description = description_element.text
            else:
                description = ""

            if news_provider == "google":
                pub_date_str = item.find("pubDate").text
            elif news_provider == "bing":
                pub_date_str = item.find("pubDate", namespace).text
            else:
                pub_date_str = item.find("pubDate").text  # For other providers

            if isinstance(pub_date_str, str):
                # Remove "GMT" from pub_date_str
                if " GMT" in pub_date_str:
                    pub_date_str = pub_date_str.replace(" GMT", "")

            pub_date = None

            try:
                # Parse the date with the timezone information
                pub_date = datetime.strptime(
                    pub_date_str, "%a, %d %b %Y %H:%M:%S %z")
            except ValueError:
                try:
                    # If the "+0000" timezone offset is missing, try parsing without timezone information
                    pub_date = datetime.strptime(
                        pub_date_str, "%a, %d %b %Y %H:%M:%S")
                except ValueError:
                    try:
                        # Try the '2023-08-18T23:05:24Z' format
                        pub_date = datetime.strptime(
                            pub_date_str, "%Y-%m-%dT%H:%M:%SZ")
                    except ValueError:
                        try:
                            # Try the '2023-08-18 23:05:24' format
                            pub_date = datetime.strptime(
                                pub_date_str, "%Y-%m-%d %H:%M:%S")
                        except ValueError as e:
                            print("Error parsing date:", e)

            if pub_date is not None:
                if pub_date.tzinfo is None:
                    # Convert pub_date to GMT/UTC using pytz
                    pub_date = gmt_timezone.localize(pub_date)

            if (datetime.now(pytz.utc) - pub_date) <= timedelta(hours=max_hours_old):
                if title:
                    # Check if the article with the same title already exists
                    if any(article["title"] == title for article in articles):
                        logger.warning(colorize_text(
                            f"Article with title '{title}' already exists. Skipping.", 'yellow'))
                        continue

                    # logger.info(f"++++++++{news_provider}++++++++++ " + title)
                    # logger.info(f"++++++++{news_provider}++++++++++ " + link)
                    # logger.info(f"++++++++{news_provider}++++++++++ " + description)
                    # logger.info(f"++++++++{news_provider}++++++++++ " + pub_date_str)

                    if title and description:
                        article = {
                            "title": title,
                            "link": link,
                            "description": description,
                            "pub_date": pub_date_str
                        }
                        # prompt += f"{len(articles)}: title: {article['title']}, published: {pub_date_str}\n"
                        # logger.info("----------------- " + prompt)
                    else:
                        article = {
                            "title": title,
                            "link": link,
                            "description": "",
                            "pub_date": pub_date_str
                        }
                    articles.append(article)
                else:
                    logger.warning(colorize_text(
                        "Skipping article due to missing title.", 'yellow'))
                    continue

                """
                if len(articles) >= max_articles:
                    logger.warning("Max articles reached")
                    break
                """
    # Extract info from marketbeat news table
    elif news_provider == "marketbeat":
        soup = BeautifulSoup(xml_content, 'html.parser')
        # Find the table with class "s-table"
        table = soup.find('table', class_='s-table')

        # Initialize lists to store titles and dates
        titles = []
        dates = []

        # Find all table rows in the table body
        rows = table.tbody.find_all('tr', class_=lambda x: x != 'bottom-sort')

        # Iterate through the rows to extract titles and dates
        for row in rows:
            columns = row.find_all('td')
            title_column = columns[1]
            title = title_column.a.get_text(strip=True)
            date_with_source = title_column.br.next_sibling.strip()

            # Adjust the date format and parsing format
            dt_format = "%B %d at %I:%M %p"
            date_str = date_with_source.split(' - ')[-1].strip()

            # Convert the date to a datetime object
            pub_date = datetime.strptime(date_str, dt_format)

            # Set the year to the current year (since it's missing)
            pub_date = pub_date.replace(year=datetime.now().year)

            # Set the timezone for pub_date to UTC
            pub_date = pub_date.replace(tzinfo=pytz.UTC)

            if (datetime.now(pytz.utc) - pub_date) <= timedelta(hours=max_hours_old):
                article = {
                    "title": title,
                    "link": "",
                    "description": "",
                    "pub_date": pub_date
                }
                articles.append(article)

    # Extract info from fxstreetpro api
    elif news_provider == "fxstreet":
        url = "https://50dev6p9k0-3.algolianet.com/1/indexes/*/queries"

        headers = {
            "Content-Type": "application/json",
            "X-Algolia-Application-Id": "50DEV6P9K0",
            "X-Algolia-API-Key": "cd2dd138c8d64f40f6d06a60508312b0",
            "X-Algolia-Agent": "Algolia for vanilla JavaScript (lite) 3.25.1;instantsearch.js 2.6.3;JS Helper 2.24.0",
        }

        referrer = "https://www.fxstreet.com/"
        referrer_policy = "strict-origin-when-cross-origin"

        # Dictionary for the query parameters
        params = {
            "hitsPerPage": 50,
            "maxValuesPerFacet": 99999,
            "page": 0,
            "filters": "CultureName:en AND (Category:'Crypto' OR Category:'Breaking Crypto News' OR Category:'Premium Crypto News' OR Category:'Crypto News' OR Category:'Crypto Analysis' OR Category:'Cryptocurrencies Sponsored News')",
            "facets": ["Tags", "AuthorName"],
            "tagFilters": "",
        }

        # Define the query
        query = {
            "requests": [
                {
                    "indexName": "FxsIndexPro",
                    "params": urllib.parse.urlencode(params),
                }
            ]
        }

        session: Session = Session()
        session.headers.update(headers)

        response = session.post(url, json=query)
        data = response.json()

        # Get the current time in UTC with timezone information
        current_time_utc = datetime.now(pytz.utc)

        # Check if the response contains 'results'
        if 'results' in data:
            results = data['results']

            # Iterate through the results
            for result in results:
                # Check if 'hits' exists in the result
                if 'hits' in result:
                    hits = result['hits']

                    # Iterate through the hits
                    for hit in hits:
                        # Extract the desired information for each hit
                        title = hit.get('Title', 'N/A')

                        # Convert the Unix timestamp to UTC datetime
                        pub_date_timestamp = hit.get('PublicationTime', 0)
                        pub_date_utc = datetime.utcfromtimestamp(
                            pub_date_timestamp).replace(tzinfo=pytz.utc)

                        link = hit.get('FullUrl', 'N/A')

                        # Calculate the age of the article in hours
                        age_in_hours = (current_time_utc -
                                        pub_date_utc).total_seconds() / 3600

                        # Check if the article is within the specified age limit
                        if age_in_hours <= max_hours_old:
                            article = {
                                "title": title,
                                "link": link,
                                "description": "",
                                "pub_date": pub_date_utc.strftime('%Y-%m-%d %H:%M:%S UTC')
                            }
                            articles.append(article)

    # Extract info from spotonchain api
    elif news_provider == "spotonchain":
        headers = {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Edg/115.0.1901.183',
        }

        num_items = 5
        url = f"https://api.spotonchain.com/api/onchain/get_latest_signals?last_publish_time=0&limit={num_items}"

        session = Session()
        session.headers.update(headers)
        response = session.get(url)

        if response.status_code == 200:
            json_data = response.text
            extracted_articles = []
            current_time_utc = datetime.now(timezone.utc)

            data = json.loads(json_data)
            latest_signals = data.get("data", {}).get("latest_signals", [])

            for item in latest_signals:
                content = item.get("content")
                title = item.get("title")  # Extracting the title
                if content:
                    publish_time = item.get("publish_time")
                    update_time = item.get("update_time")
                    # Parse the content string as JSON
                    content = json.loads(content)
                    extracted_text = ""
                    for content_item in content:
                        if content_item.get("type") == "paragraph" or content_item.get("type") == "bulleted-list":
                            for child in content_item.get("children", []):
                                if 'text' in child:
                                    extracted_text += child['text']
                        # elif content_item.get("type") == "image" and content_item.get("url"):
                        #    extracted_text += f"Fig caption: {content_item['url']}\n"
                    extracted_articles.append({
                        "title": title,  # Adding the title to the article data
                        "description": extracted_text,
                        "pub_date": publish_time,
                        "update_time": update_time
                    })

            articles = []
            for article_data in extracted_articles:
                # Convert the Unix timestamp to UTC datetime
                pub_date_timestamp = article_data["pub_date"]
                pub_date_utc = datetime.utcfromtimestamp(
                    pub_date_timestamp).replace(tzinfo=pytz.utc)

                # Calculate the age of the article in hours
                age_in_hours = (current_time_utc -
                                pub_date_utc).total_seconds() / 3600

                # Check if the article is within the specified age limit
                if age_in_hours <= max_hours_old:
                    article = {
                        # Including the title in the final article
                        "title": article_data["title"],
                        "link": "",
                        "description": article_data["description"],
                        "pub_date": pub_date_utc.strftime('%Y-%m-%d %H:%M:%S UTC'),
                        # "update_time": article_data["update_time"]
                    }
                    articles.append(article)

    # Extract info signals from santiment api
    elif news_provider == "santiment":
        headers = {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Edg/115.0.1901.183',
        }

        num_items = 100
        url = f'https://sanr-l2-api.production.internal.santiment.net/api/v1/leaderboards/forecasts?filter="status":{{"$ne":"closed"}}&sort=createdAt:desc,signalID:asc&take={num_items}'

        session = Session()
        session.headers.update(headers)
        response = session.get(url)

        if response.status_code == 200:
            json_data = response.text

            # Load JSON data into a dictionary
            data = json.loads(json_data)

            # Current datetime
            current_time = datetime.now(timezone.utc)

            # List of unique fields to identify duplicates
            unique_fields = ["id", "signalID"]

            # "totalMedalsAmount" # User should have 5 or more
            # "totalQilinMedalsAmount" # Qilin Highest average positive performance - good for direction up
            # "totalDragonMedalsAmount" # Dragon Highest total positive performance - good for direction up
            # "totalPhoenixMedalsAmount" # Phoenix Highest total negative performance - good for direction down

            # Filter records with totalMedalsAmount >= 5 and createdAt less than max_hours_old
            filtered_json_data = [
                record for record in data['data']
                if 'performance' in record and  # Check if 'performance' key exists in the record
                (record['issuer']['totalMedalsAmount'] >= 5) and
                (current_time - datetime.fromisoformat(record['createdAt'].replace('Z', '+00:00')).replace(tzinfo=timezone.utc)) < timedelta(hours=max_hours_old) and
                ((record['direction'] == 'up' and (record['issuer']['totalQilinMedalsAmount'] >= 1 or record['issuer']['totalDragonMedalsAmount'] >= 1)) or
                 (record['direction'] == 'down' and record['issuer']['totalPhoenixMedalsAmount'] >= 1))
            ]

            # Remove records with identical id and signalID
            seen_records = set()
            unique_filtered_data = []

            for record in filtered_json_data:
                record_id = tuple(record[field] for field in unique_fields)
                if record_id not in seen_records:
                    seen_records.add(record_id)
                    unique_filtered_data.append(record)

            # Group by fields excluding the unique ones
            grouped_data = {}
            group_criteria = ["symbol", "direction",
                              "status"]  # Criteria for grouping

            for record in unique_filtered_data:
                group_key = tuple(record[field] for field in group_criteria)
                grouped_data.setdefault(group_key, []).append(record)

            # Find common fields among grouped records
            if grouped_data:  # Check if grouped_data is not empty
                common_fields = set(
                    next(iter(grouped_data.values()))[0].keys())
                for group_key, group_records in grouped_data.items():
                    common_fields &= set(group_records[0].keys())

                    for record in group_records:
                        common_fields &= set(record.keys())
            else:
                common_fields = set()  # Or any other appropriate default value if grouped_data is empty

            # Keep only the fields that have the same values across records within each group
            final_data = []

            for group_records in grouped_data.values():
                common_record = {}
                for field in common_fields:
                    values = [record[field] for record in group_records]
                    if all(isinstance(value, (int, float, str)) and value == values[0] for value in values):
                        common_record[field] = values[0]

                final_data.append(common_record)

            articles = []
            for record in final_data:
                if "openDate" in record:
                    # Replace "openDate" with the correct key name if needed
                    pub_date_str = record["openDate"]
                    pub_date_utc = datetime.strptime(
                        pub_date_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=pytz.utc)

                    if record["direction"] == "up":
                        sentiment = "POSITIVE"
                    elif record["direction"] == "down":
                        sentiment = "NEGATIVE"

                    title = "Santiment " + \
                        record["direction"] + " signal for " + \
                            record["symbol"] + " - " + sentiment
                    description = ""
                    """
                    description = (f"Performance: {record.get('performance', 'N/A')}\n"
                                   f"Lifetime: {record.get('lifetime', 'N/A')}\n"
                                   f"Min Price: {record.get('minPrice', 'N/A'):.10f}\n"
                                   f"Max Price: {record.get('maxPrice', 'N/A'):.10f}\n"
                                   f"Close Price: {record.get('closePrice', 'N/A'):.10f}\n"
                                   f"Take Profit Price: {record.get('takeProfitPrice', 'N/A')}\n"
                                   f"Open Price: {record.get('openPrice', 'N/A'):.10f}\n")
                    """
                    signal = {
                        "title": title,
                        "description": description,
                        "pub_date": pub_date_utc.strftime('%Y-%m-%d %H:%M:%S UTC'),
                        "update_time": pub_date_utc.strftime('%Y-%m-%d %H:%M:%S UTC')
                    }
                    articles.append(signal)

                else:
                    # Handle the case when the key "openDate" is missing or has a different name in the JSON data
                    # You can set a default value for pub_date_str or handle it according to your requirements
                    pass  # or assign a default value, raise an error, or handle it based on your logic

    return articles


def fetch_asknews_sentiment(token_name, max_hours_old, asknews_api_key, asknews_tokens):
    # API Endpoint and Headers
    url = 'https://api.asknews.app/v1/sentiment'
    headers = {
        'accept': 'application/json',
        'Authorization': f'Bearer {asknews_api_key}',
        'Content-Type': 'application/json'
    }

    session = Session()
    session.headers.update(headers)

    articles = []
    sentiment_headline = ""
    # ["bitcoin", "ethereum", "cardano", "uniswap", "ripple", "solana", "polkadot", "polygon", "chainlink", "tether"]
    token = token_name.lower()
    if token in asknews_tokens:
        positive_metric = "news_positive"
        negative_metric = "news_negative"
        extended_hours = 4

        # Current datetime
        current_time = datetime.now(timezone.utc)
        date_to = current_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        date_from = (current_time - timedelta(hours=extended_hours)
                     ).strftime('%Y-%m-%dT%H:%M:%SZ')

        # Data payload for the POST requests - Extended time period for positive and negative sentiment
        positive_extended_data = {
            "slug": token,
            "metric": positive_metric,
            "date_from": date_from,
            "date_to": date_to
        }

        negative_extended_data = {
            "slug": token,
            "metric": negative_metric,
            "date_from": date_from,
            "date_to": date_to
        }

        # Making the POST requests for the extended time period for both positive and negative sentiment
        positive_extended_response = requests.post(
            url, headers=headers, json=positive_extended_data)
        negative_extended_response = requests.post(
            url, headers=headers, json=negative_extended_data)

        if positive_extended_response.status_code == 200 and negative_extended_response.status_code == 200:
            # Extracting timeseries data for extended time period for positive and negative sentiment
            positive_extended_sentiment_data = positive_extended_response.json()[
                'data']['getMetric']['timeseriesData']
            negative_extended_sentiment_data = negative_extended_response.json()[
                'data']['getMetric']['timeseriesData']

            # Extracting values for z-score calculation for positive and negative sentiment
            positive_extended_values = [data_point['value']
                                        for data_point in positive_extended_sentiment_data]
            negative_extended_values = [data_point['value']
                                        for data_point in negative_extended_sentiment_data]

            # Calculate mean and standard deviation for z-score calculation for positive and negative sentiment
            positive_mean = statistics.mean(positive_extended_values)
            positive_std_dev = statistics.stdev(positive_extended_values) if len(
                positive_extended_values) > 1 else 0

            negative_mean = statistics.mean(negative_extended_values)
            negative_std_dev = statistics.stdev(negative_extended_values) if len(
                negative_extended_values) > 1 else 0

            # Data payload for the POST requests - Positive and Negative Sentiment for recent time window
            recent_date_from = (
                current_time - timedelta(hours=max_hours_old)).strftime('%Y-%m-%dT%H:%M:%SZ')
            positive_recent_data = {
                "slug": token,
                "metric": positive_metric,
                "date_from": recent_date_from,
                "date_to": date_to
            }
            negative_recent_data = {
                "slug": token,
                "metric": negative_metric,
                "date_from": recent_date_from,
                "date_to": date_to
            }

            # Making the POST requests for the recent time window for both positive and negative sentiment
            positive_recent_response = requests.post(
                url, headers=headers, json=positive_recent_data)
            negative_recent_response = requests.post(
                url, headers=headers, json=negative_recent_data)

            if positive_recent_response.status_code == 200 and negative_recent_response.status_code == 200:
                # Extracting timeseries data for recent time window for positive and negative sentiment
                positive_sentiment_data = positive_recent_response.json(
                )['data']['getMetric']['timeseriesData']
                negative_sentiment_data = negative_recent_response.json(
                )['data']['getMetric']['timeseriesData']

                # Extracting values for z-score calculation for recent positive and negative sentiment
                recent_positive_values = [data_point['value']
                                          for data_point in positive_sentiment_data]
                recent_negative_values = [data_point['value']
                                          for data_point in negative_sentiment_data]

                # Calculate z-scores for the recent time window for positive and negative sentiment
                if positive_std_dev != 0:
                    z_scores_positive = [
                        (value - positive_mean) / positive_std_dev for value in recent_positive_values]
                else:
                    # All z-scores are 0 if there is no variation
                    z_scores_positive = [0 for _ in recent_positive_values]

                if negative_std_dev != 0:
                    z_scores_negative = [
                        (value - negative_mean) / negative_std_dev for value in recent_negative_values]
                else:
                    # All z-scores are 0 if there is no variation
                    z_scores_negative = [0 for _ in recent_negative_values]

                # Check if the z-scores for the recent period are above the average for positive and negative sentiment
                average_z_score_positive = statistics.mean(
                    z_scores_positive) if z_scores_positive else 0
                average_z_score_negative = statistics.mean(
                    z_scores_negative) if z_scores_negative else 0

                # Check if the z-scores for the recent period are above average for positive and negative sentiment
                if average_z_score_positive > 0:  # You can set a threshold for positivity/negativity
                    recommendation_positive = "Positive sentiment is higher than average"
                else:
                    recommendation_positive = "Positive sentiment is not higher than average"

                if average_z_score_negative > 0:  # You can set a threshold for positivity/negativity
                    recommendation_negative = "Negative sentiment is higher than average"
                else:
                    recommendation_negative = "Negative sentiment is not higher than average"

                # print(f"Average Z-Score Positive: {average_z_score_positive}")
                # print(f"Recommendation Positive: {recommendation_positive}")

                # print(f"Average Z-Score Negative: {average_z_score_negative}")
                # print(f"Recommendation Negative: {recommendation_negative}")

                # Perform further analysis or trading recommendation based on z-scores and sentiment trends for both positive and negative sentiment

                # Generating the 'prompt' variable with meaningful sentiment analysis information
                prompt = f"""Ask News conducted a sentiment analysis of news articles over a {extended_hours}-hour period and a more recent {max_hours_old}-hour period. The analysis shows that:
Positive Sentiment: The average positive sentiment observed during the recent {max_hours_old}-hour period is {'lower than' if average_z_score_positive < 0 else 'higher than or equal to'} the average of the {extended_hours}-hour period, indicated by a {'negative' if average_z_score_positive < 0 else 'positive'} Z-score ({average_z_score_positive}). This suggests that the overall positive sentiment in news articles is {'not higher than' if average_z_score_positive < 0 else 'higher than or equal to'} the average observed during this time.
Negative Sentiment: The average negative sentiment observed during the recent {max_hours_old}-hour period is {'higher than' if average_z_score_negative > 0 else 'lower than or equal to'} the average of the {extended_hours}-hour period, indicated by a {'positive' if average_z_score_negative > 0 else 'negative'} Z-score ({average_z_score_negative}). This suggests that the negative sentiment in news articles is {'higher than' if average_z_score_negative > 0 else 'lower than or equal to'} the average observed during this period."""

                # print(prompt)  # Displaying the generated prompt
                current_time = datetime.now(timezone.utc)
                sentiment = {
                    "title": "Ask News",
                    "description": prompt,
                    "pub_date": current_time.strftime('%Y-%m-%d %H:%M:%S UTC'),
                    "update_time": current_time.strftime('%Y-%m-%d %H:%M:%S UTC')
                }
                articles.append(sentiment)
                # print(prompt)  # Displaying the generated prompt
                logger.info("++++++++ " + colorize_text("Found:",
                            'green') + " " + prompt)

                sentiment_headline += f"Title: {prompt} Updated: {current_time.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"

            else:
                logger.warning(
                    f"Error fetching sentiment data for {token}. Status codes: POS:{positive_response.status_code} NEG:{negative_response.status_code}")

        else:
            logger.warning(
                f"Error fetching sentiment data for {token}. Status codes: POS:{positive_response.status_code} NEG:{negative_response.status_code}")

    return articles, sentiment_headline


def fetch_subreddits_containing(keyword, reddit_client):
    # Search for subreddits containing the keyword
    crypto_subreddits = []
    for subreddit in reddit_client.subreddits.search_by_name(keyword, exact=False):
        crypto_subreddits.append(subreddit.display_name)
    return crypto_subreddits


def fetch_reddit_posts(tokens, max_hours_old, reddit_client_id, reddit_client_secret):
    min_upvotes = 0
    min_karma = 1000

    post_number = 1  # Initialize the post number

    # Set up the Reddit client
    reddit = praw.Reddit(
        client_id=reddit_client_id,
        client_secret=reddit_client_secret,
        user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Edg/115.0.1901.200'
    )

    logger.info("++++++++ " + colorize_text("Checking Reddit:", 'magenta'))

    # Calculate timestamp
    twenty_four_hours_ago = time.time() - max_hours_old * 60 * 60

    additional_keywords = "news"
    # search_terms = [f"${tokens[0]}", tokens[0], tokens[1], additional_keywords]
    search_terms = [f"${tokens[0]}", tokens[0], tokens[1]]

    # Fetch list of subreddits containing the word 'crypto' once
    subreddits_to_search = fetch_subreddits_containing('crypto', reddit)

    fetched_posts = []
    seen_posts = set()  # Maintain a set to track post IDs and avoid duplicates

    query_string = " OR ".join(search_terms)
    for subreddit_name in subreddits_to_search:
        results = reddit.subreddit(subreddit_name).search(
            query_string, time_filter='day', limit=1000)
        for post in results:
            if post.id in seen_posts:
                continue
            try:
                if (post.created_utc > twenty_four_hours_ago and
                    post.author.comment_karma + post.author.link_karma > min_karma and
                        post.score >= min_upvotes):
                    # Ensure the post title contains at least one of the search terms
                    if any(t.lower() in post.title.lower() for t in [tokens[0], f"${tokens[0]}", tokens[1]]):
                        # Mark this post ID as processed
                        seen_posts.add(post.id)
                        fetched_posts.append(post)
            # Handle if the author or their karma attributes are missing
            except AttributeError:
                continue
            # Catch other general exceptions
            except Exception as e:
                logger.warning(
                    "++++++++ " + colorize_text(f"Error processing post {post.id}: {e}", 'yellow'))
                continue

    prompt = "\nPosts:\n"
    posts = []

    for post in fetched_posts:
        post_date = datetime.utcfromtimestamp(
            post.created_utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        post_content = replace_non_bmp_characters(post.title)
        post_body = replace_non_bmp_characters(post.selftext)
        upvotes = post.score
        author = post.author.name if post.author else 'N/A'

        post_entry = {
            "date": post_date,
            "title": post_content,
            "body": post_body,
            "upvotes": upvotes,
            "author": author
        }

        posts.append(post_entry)

        logger.info("++++++++ " + colorize_text("Found:",
                    'green') + " " + post_content)

        post_entry_str = f"{post_number}: Post on {post_date}: {post_content}, Upvotes: {upvotes}, Author: {author}\n"
        prompt += post_entry_str
        post_number += 1

    return posts, prompt, post_number


def get_coinmarketcap_gravity_followers_count(guid, session):
    url = "https://api-gravity.coinmarketcap.com/gravity/v3/gravity/profile/query"
    data = {
        "guid": guid
    }
    response = session.options(url)

    response = session.post(url, json=data)
    followers_data = response.json()
    followers = followers_data.get('data', {}).get(
        'gravityAccount', {}).get('followers')
    return followers


def fetch_coinmarketcap_gravity_posts(tokens, max_hours_old):
    post_number = 1  # Initialize the post number
    min_followers = 1000

    # Create a session
    session = requests.Session()

    base_url = "https://api-gravity.coinmarketcap.com/gravity/v3/gravity/search"

    # Combine tokens with additional keywords
    additional_keywords = "news"
    search_terms = [f"${token}" for token in tokens] + \
        tokens + [additional_keywords]

    # Create the full URL with search terms
    full_url = base_url + "?" + "&".join([f"word={term}" for term in search_terms]) + \
        f"&start=1&fullWord={'+'.join(search_terms)}&handleOnly=false&languageCode=en&latestSort=true"

    # Headers for requests
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/117.0",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-AU",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache",
        "referrer": "https://coinmarketcap.com/community/"
    }

    # Options request
    session.options(full_url, headers=headers)

    # Calculate the datetime for the time threshold
    current_utc_time = datetime.utcnow()
    time_threshold = current_utc_time - timedelta(hours=max_hours_old)

    logger.info(
        "++++++++ " + colorize_text("Checking CoinMarketCap Community:", 'magenta'))

    # Use the session to make the request
    response = session.get(full_url, headers=headers)
    data = response.json()

    fetched_posts = []

    # Extracting data from the JSON
    if 'data' in data and isinstance(data['data'], list) and data['data']:
        for post in data['data']:
            created_time_milliseconds = int(post['postTime'])
            created_time_utc = datetime.utcfromtimestamp(
                created_time_milliseconds / 1000.0)
            if created_time_utc >= time_threshold:
                guid = post['owner']['guid']
                followers_count = get_coinmarketcap_gravity_followers_count(
                    guid, session)
                if followers_count is not None:
                    if int(followers_count) >= min_followers:
                        gravity_id = post['gravityId']
                        nickname = post['owner']['nickname']
                        handle = post['owner']['handle']
                        avatar_url = post['owner']['avatar']['url']
                        if 'textContent' in post:
                            content = replace_non_bmp_characters(
                                post['textContent'])
                        else:
                            content = ""
                        like_count = int(post['likeCount'])
                        comment_count = int(post['commentCount'])

                        # Check if the token or token name exists in the content
                        if any(token.lower() in content.lower() for token in tokens):
                            # Add the post information to the fetched_posts list
                            post_entry = {
                                "Gravity ID": gravity_id,
                                "Nickname": nickname,
                                "Handle": handle,
                                "Avatar URL": avatar_url,
                                "Content": content,
                                "Like Count": like_count,
                                "Comment Count": comment_count,
                                "Created Time (UTC)": created_time_utc,
                                "Number of followers": followers_count if followers_count is not None else "Follower count not available"
                            }
                            fetched_posts.append(post_entry)
    else:
        print("!!! DIFFERENT !!!\n", data)

    session.close()

    # Create the prompt and return values
    prompt = "\nPosts:\n"
    for post_entry in fetched_posts:
        prompt += f"{post_number}: Post on {post_entry['Created Time (UTC)']}: {post_entry['Content']}, Like Count: {post_entry['Like Count']}, Author: {post_entry['Nickname']}\n"
        logger.info("++++++++ " + colorize_text("Found:", 'green') + " " +
                    f"{post_number}: Post on {post_entry['Created Time (UTC)']}: {post_entry['Content']}, Like Count: {post_entry['Like Count']}, Author: {post_entry['Nickname']}\n")
        post_number += 1

    return fetched_posts, prompt, post_number


def set_coin_dict(self, context):
    pair_whitelist = self.config.get("exchange")['pair_whitelist']

    # Check if pair_whitelist is empty
    if not pair_whitelist:
        logger.error(
            "Error: pair_whitelist is empty. You must provide a non-empty static whitelist.")
    else:
        # Check for patterns like ".*" in pair_whitelist
        invalid_patterns = [pair for pair in pair_whitelist if ".*" in pair]
        if invalid_patterns:
            logger.error(
                "Error: pair_whitelist must be static not dynamic. The following patterns are not allowed in static a pair_whitelist: " + ", ".join(invalid_patterns))
        else:
            spot_dict = {}
            for pair in pair_whitelist:
                if context == "futures":
                    pair = pair.replace('USDT:USDT', 'USDT')

                # Split the pair by '/' to separate the tokens
                tokens = pair.split('/')

                # Generate the value string based on the tokens
                value = f"{tokens[0]} {tokens[1]} crypto"

                # Add the entry to the dictionary
                spot_dict[pair] = value

            # Print the resulting spot_dict
            print(spot_dict)

    futures_suffix = ":USDT"

    if context == "spot":
        coin_dict = spot_dict
    elif context == "futures":
        coin_dict = {key + futures_suffix: value for key,
                     value in spot_dict.items()}
    else:
        raise ValueError(f"Unknown context: {context}")

    return coin_dict
