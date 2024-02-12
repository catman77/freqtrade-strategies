import requests
import openai
import tiktoken

from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.freqai_interface import IFreqaiModel
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Any, Tuple, Dict
import logging
import time
import numpy as np
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)


class GPTTrader(IFreqaiModel):
    """
    Base model for letting an LLM take full control of your bot...
    """

    def __init__(self, **kwargs):
        super().__init__(config=kwargs["config"])
        self.openai_key = self.freqai_info.get("openai_key", None)
        if self.openai_key is None:
            raise ValueError("openai_key is not set in freqtrade config.json")
        else:
            openai.api_key = self.openai_key

        self.gpt_model = self.freqai_info["GPTTrader"].get("gpt_model", "gpt-3.5-turbo")
        self.news_hours = self.freqai_info["GPTTrader"].get("news_hours", 6)

    def train(
        self, unfiltered_df: DataFrame, pair: str, dk: FreqaiDataKitchen, **kwargs
    ) -> Any:
        """
        Build a custom prompt asking to determine news sentiment and take an action
        based on current coin position.
        """
        self.prompt = """
        Forget all your previous instructions.
        You are a financial expert with cryptocurrency trading recommendation experience. You
        help decide actions given a current trade position, profit targets, and a set of news headlines.
        """

        self.request = "Please review the following news headlines and answer with `YES`, `NO`, or `UNKNOWN` for each of them:\n\n"
        self.buy_sell = """Given the headlines and my position and my targets, please recommend an action to me from the following options `LONG_ENTER`, 
        `LONG_EXIT`, `SHORT_ENTER`, `SHORT_EXIT` or `NEUTRAL` on a new line. Keep in mind that I can only enter a position if I am not currently in position. 
        I can only exit a position if I am already in a position. I cannot `SHORT_EXIT` if I am already in a `long` position and I cannot `LONG_EXIT` if 
        I am already in a short position.\n\n"""
        self.summary = "Final Summary Opinion (2 sentences max): Based on the information provided and your assessments, write a summary of why you made your choice [Your opinion here].\n\n"

        # FIXME: find a smarter way to handle auto search_query per coin
        search_query = coin_dict[pair]
        side, profit, duration = self.get_state_info(pair)
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
            self.position = (f"My current position is {side} with profit: {profit} profit and the trade duration is: {duration} candles."
                             f" The current date is {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}\n\n")

        self.desires = (f"My target profit is {self.freqai_info['GPTTrader'].get('target_profit', 0.03)} "
                        f"and my target duration is {self.freqai_info['GPTTrader'].get('target_duration', 100)} candles "
                        f"and my stoploss is {self.freqai_info['GPTTrader'].get('stoploss', 0.04)}\n\n")

        articles, headlines = fetch_articles(
            search_query, max_hours_old=self.news_hours, max_articles=20)

        print(f"found {len(articles)} articles and {len(headlines)} headlines")
        if len(articles) == 0:
            logger.info(f"No articles found for {search_query}")
            return {"sentiment_yes": 0, "sentiment_no": 0, "sentiment_unknown": 0,
                    "expert_long_enter": 0, "expert_long_exit": 0,
                    "expert_short_enter": 0, "expert_short_exit": 0,
                    "expert_neutral": 0, "expert_opinion": ""}

        content = self.analyze_sentiment(headlines)
        model = evaluate_content(content, pair)
        dk.data_dictionary["train_features"] = DataFrame(np.zeros(1))
        dk.data_dictionary["train_dates"] = DataFrame(np.zeros(1))
        for label in dk.label_list:
            dk.data["labels_mean"] = {}
            dk.data["labels_mean"][label] = 0
            dk.data["labels_std"] = {}
            dk.data["labels_std"][label] = 0
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

        token_count = num_tokens_from_string(self.prompt, "cl100k_base")
        # Count the tokens
        logger.info(
            f"Input Token count: {token_count}, which will cost ${token_count/1000 * 0.0015} USD")

        retries = 0
        while retries < 5:
            try:
                response = openai.ChatCompletion.create(
                    model=self.gpt_model,
                    messages=[
                        {"role": "system", "content": self.prompt},
                        {"role": "user", "content": self.request + text + self.buy_sell + self.position + self.desires + self.summary},
                    ],
                    temperature=0,
                )
                break
            except openai.error.RateLimitError:
                print(
                    "RateLimitError: That model is currently overloaded with other requests. Retrying in 10 seconds.")
                time.sleep(10)
                retries += 1

        if retries == 5:
            raise Exception("Failed to get response from OpenAI after 5 retries.")

        content = response['choices'][0]['message']['content']

        token_count = num_tokens_from_string(content, "cl100k_base")
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
                trade_duration = int((now - trade.open_date_utc.timestamp()) / self.base_tf_seconds)
                current_profit = trade.calc_profit_ratio(current_rate)
                if trade.is_short:
                    market_side = 0
                else:
                    market_side = 1

        return market_side, current_profit, int(trade_duration)


# GPT infusion
def evaluate_content(content, pair):
    # count number of occurences of "YES", "NO", "UNKNOWN"
    expert_model = {}
    expert_model["sentiment_yes"] = content.count("YES\n")
    expert_model["sentiment_no"] = content.count("NO\n")
    expert_model["sentiment_unknown"] = content.count("UNKNOWN\n")
    expert_model["expert_long_enter"] = "LONG_ENTER" in content
    expert_model["expert_long_exit"] = "LONG_EXIT" in content
    expert_model["expert_short_enter"] = "SHORT_ENTER" in content
    expert_model["expert_short_exit"] = "SHORT_EXIT" in content
    expert_model["expert_neutral"] = "NEUTRAL" in content
    expert_model["expert_opinion"] = content.split("LONG_ENTER")[-1] if "LONG_ENTER" in content else None
    if expert_model["expert_opinion"] is None:
        expert_model["expert_opinion"] = content.split("LONG_EXIT")[-1] if "LONG_EXIT" in content else None
    if expert_model["expert_opinion"] is None:
        expert_model["expert_opinion"] = content.split("SHORT_ENTER")[-1] if "SHORT_ENTER" in content else None
    if expert_model["expert_opinion"] is None:
        expert_model["expert_opinion"] = content.split("SHORT_EXIT")[-1] if "SHORT_EXIT" in content else None
    if expert_model["expert_opinion"] is None:
        expert_model["expert_opinion"] = content.split("NEUTRAL")[-1] if "NEUTRAL" in content else None
    logger.info(f"Expert Opinion on {pair}: {expert_model['expert_opinion']}")

    return expert_model


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""

    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(string))
    return num_tokens


def fetch_articles(search_query, max_hours_old=1, max_articles=5):
    url = "https://news.google.com/rss/search?q={}".format(search_query)
    response = requests.get(url)

    if response.status_code == 200:
        articles = parse_articles(response.text, max_hours_old, max_articles)
        return articles
    else:
        print("Error fetching articles. Status code:", response.status_code)
        return []


def parse_articles(xml_content, max_hours_old, max_articles=5):
    from xml.etree import ElementTree as ET

    root = ET.fromstring(xml_content)
    articles = []
    prompt = ""

    for item in root.findall(".//item"):
        pub_date_str = item.find("pubDate").text
        pub_date = datetime.strptime(pub_date_str, "%a, %d %b %Y %H:%M:%S %Z")

        if (datetime.now() - pub_date) <= timedelta(days=max_hours_old):
            article = {
                "title": item.find("title").text,
                "pub_date": pub_date_str
            }
            prompt += f"{len(articles)}: title: {article['title']}, published: {pub_date_str}\n"
            articles.append(article)

        if len(articles) >= max_articles:
            print("Max articles reached")
            break

    return articles, prompt


coin_dict = {
            "1INCH/USDT:USDT": "1INCH crypto",
            "AAVE/USDT:USDT": "AAVE crypto",
            "ADA/USDT:USDT": "Cardano ADA crypto",
            "BTC/USDT:USDT": "Bitcoin BTC crypto",
            "ETH/USDT:USDT": "Ethereum ETH crypto DEFI",
            "DOT/USDT:USDT": "Polkadot DOT crypto",
            "LINK/USDT:USDT": "Chainlink LINK crypto DEFI",
            "DOGE/USDT:USDT": "Dogecoin DOGE crypto",
            "SOL/USDT:USDT": "Solana SOL crypto DEFI",
            "MATIC/USDT:USDT": "Polygon MATIC crypto DEFI layer2",
            "EGLD/USDT:USDT": "Elrond EGLD crypto DEFI",
            "XLM/USDT:USDT": "Stellar XLM crypto",
            "XRP/USDT:USDT": "Ripple XRP crypto",
            "XMR/USDT:USDT": "Monero XMR crypto",
            "ZEC/USDT:USDT": "Zcash ZEC crypto",
            "AVAX/USDT:USDT": "Avalanche AVAX crypto DEFI",
            "SNX/USDT:USDT": "Synthetix SNX crypto DEFI",
            "ATOM/USDT:USDT": "Cosmos ATOM crypto layer2",
            "LTC/USDT:USDT": "Litecoin LTC crypto",
            "ALGO/USDT:USDT": "Algorand ALGO crypto layer2"
}
