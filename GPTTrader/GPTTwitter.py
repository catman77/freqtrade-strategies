import requests
import openai
import tiktoken

from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.freqai_interface import IFreqaiModel
from datetime import datetime, timedelta, timezone
import pytz
import re
from requests import Session
from pandas import DataFrame
from typing import Any, Tuple, Dict
import logging
import time
import numpy as np
from freqtrade.persistence import Trade

# from Perplexity import Perplexity

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


class GPTTwitter(IFreqaiModel):
    """
    Base model for letting an LLM take full control of your bot...
    """

    def __init__(self, **kwargs):
        super().__init__(config=kwargs["config"])

        # If you prefer to use Perplextiy, uncomment the perplexity import
        # above and the line below
        self.useperplexity = False
        # self.perplexity = Perplexity()

        self.openai_key = self.freqai_info.get("openai_key", None)
        if self.openai_key is None and self.useperplexity == False:
            raise ValueError("openai_key is not set in freqtrade config.json")
        else:
            openai.api_key = self.openai_key

        self.gpt_model = self.freqai_info["GPTTrader"].get(
            "gpt_model", "gpt-3.5-turbo")
        self.news_hours = self.freqai_info["GPTTrader"].get("news_hours", 6)

        self.twitterapi_bearer_token = self.freqai_info['GPTTrader'].get(
            'twitterapi_bearer_token', "")

        # Options for setting coin_dict tradingmode for spot or futures
        tradingmode = self.config.get("trading_mode", "spot")
        self.coin_dict = set_coin_dict(tradingmode)

        # Get the list of pairs for training
        self.pairs = list(self.coin_dict.keys())

        self.news_providers = {
            "google": "https://news.google.com/rss/search?q={}",
            "bing": "https://www.bing.com/news/search?q={}&format=rss&qft=interval%3d4+sortbydate%3d1&form=PTFTNR",
        }

    def train(
        self, unfiltered_df: DataFrame, pair: str, dk: FreqaiDataKitchen, **kwargs
    ) -> Any:

        search_query = self.coin_dict[pair]
        token = pair.split("/")[0]
        token_name = extract_token_name(token, search_query)

        """Build a custom prompt to determine news sentiment and provide an action based on the current coin position."""

        self.prompt = (f"Analyze sentiment from news or tweets about {token_name} ({token}). "
                       "Your response should include a clear trading recommendation under 'Recommended Action:'.")

        self.request = ("""For each news headline or tweet, classify its sentiment as `NEGATIVE`, `NEUTRAL`, or `POSITIVE`. """
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
                              else f"and I am at a loss of {abs(current_profit_percentage)}%.")
            self.position = (f"My current position is {side}, {gain_loss_text}. The trade has been open for {duration} candles (" +
                             str(int(round(self.freqai_info['live_retrain_hours'] * 60 / 1, 0))) +
                             f"min each), and the current date is {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}.\n\n"
                             f"My target duration for the trade is {self.freqai_info['GPTTrader'].get('target_duration_candles', 100)} candles, with each candle representing {int(round(self.freqai_info['live_retrain_hours'] * 60 / 1, 0))} minutes, and I have set a stop-loss at {stoploss_percentage}%.\n\n")

        self.desires = (f"My target profit is {target_profit_percentage}% "
                        f"and my target duration is {self.freqai_info['GPTTrader'].get('target_duration_candles', 100)} candles, " +
                        "each candle is " +
                        str(int(
                            round(self.freqai_info['live_retrain_hours'] * 60 / 1, 0))) + "mins "
                        f"and my stop-loss is {stoploss_percentage}%\n\n")

        logger.info("++++ " + colorize_text("Pair: " + pair, 'green') + " ++++")

        tweets, tweets_prompt, tweet_count = fetch_tweets(
            tokens=[token, token_name], max_hours_old=self.news_hours, bearer_token=self.twitterapi_bearer_token)
        articles, headlines, headline_count = fetch_articles(
            search_query, news_providers=self.news_providers, max_hours_old=self.news_hours,
            max_articles=20, tokens=[token, token_name])

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

        if len(articles) == 0 and len(tweets) == 0:
            logger.info(f"No articles or tweets found for {search_query}")
            return {"sentiment_yes": 0, "sentiment_no": 0, "sentiment_unknown": 0,
                    "expert_long_enter": 0, "expert_long_exit": 0,
                    "expert_short_enter": 0, "expert_short_exit": 0,
                    "expert_neutral": 0, "expert_opinion": ""}

        content = ""
        if len(articles) != 0 and len(tweets) == 0:
            content = self.analyze_sentiment(headlines + "\n")
        elif len(articles) == 0 and len(tweets) != 0:
            content = self.analyze_sentiment(tweets_prompt + "\n")
        else:
            content = self.analyze_sentiment(tweets_prompt + headlines + "\n")

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

        # Using Perplexity
        if self.useperplexity == True:
            logger.info("++++ " + colorize_text("Using Perplexity...", 'cyan'))
            logger.info("++++ " + colorize_text("Analysing Sentiment:", 'green') + " ++++ " + str(self.prompt) + ": " +
                        str(self.request) + str(text) + str(self.buy_sell) + str(self.position) + str(self.desires) + str(self.summary))

            response = self.perplexity.search(str(self.prompt) + ": " + str(self.request) + str(
                text) + str(self.buy_sell) + str(self.position) + str(self.desires) + str(self.summary))
            content = response
            logger.info("++++ " + colorize_text("Sentiment Response:",
                                                'green') + " ++++ " + content)

            return content

        # Using OpenAI GPT
        else:
            logger.info("++++ " + colorize_text("Using OpenAI...", 'cyan'))
            token_count = num_tokens_from_string(self.prompt, "cl100k_base")
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
                    response = openai.ChatCompletion.create(
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


def colorize_text(text, text_color=None, bg_color=None):
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

    if text_color and text_color.lower() in color_dict:
        color_code += f'\033[1;3{color_dict[text_color.lower()]}m'

    if bg_color and bg_color.lower() in color_dict:
        color_code += f'\033[1;4{color_dict[bg_color.lower()]}m'

    if color_code != '':
        # resets color to terminal default
        color_text = color_code + text + '\033[0m'

    return color_text


def evaluate_content(content, pair):
    # count number of occurrences of "POSITIVE", "NEGATIVE", "NEUTRAL"
    expert_model = {}
    expert_model["sentiment_yes"] = content.count("POSITIVE\n")
    expert_model["sentiment_no"] = content.count("NEGATIVE\n")
    expert_model["sentiment_unknown"] = content.count(
        "NEUTRAL\n") - content.count("Recommended Action: NEUTRAL\n")

    # Check for trading recommendations based on the expected format
    expert_model["expert_long_enter"] = "Recommended Action: LONG_ENTER\n" in content
    expert_model["expert_long_exit"] = "Recommended Action: LONG_EXIT\n" in content
    expert_model["expert_short_enter"] = "Recommended Action: SHORT_ENTER\n" in content
    expert_model["expert_short_exit"] = "Recommended Action: SHORT_EXIT\n" in content
    expert_model["expert_neutral"] = "Recommended Action: NEUTRAL\n" in content

    keywords = ["LONG_ENTER", "LONG_EXIT",
                "SHORT_ENTER", "SHORT_EXIT", "NEUTRAL"]
    keyword_found = any("Recommended Action: " + keyword +
                        "\n" in content for keyword in keywords)
    expert_model["expert_opinion"] = None

    if keyword_found:
        # Look for the "Final Summary Opinion:" if a keyword is found
        final_summary_start = content.find("Final Summary Opinion:")
        if final_summary_start != -1:
            expert_model["expert_opinion"] = content[final_summary_start +
                                                     len("Final Summary Opinion:"):].strip()
        else:
            for keyword in keywords:
                if "Recommended Action: " + keyword + "\n" in content:
                    split_content = content.split(
                        "Recommended Action: " + keyword)

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


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""

    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(string))
    return num_tokens


def fetch_articles(search_query, news_providers, max_hours_old, max_articles, tokens):
    articles = []
    prompt = "\nThe following news headlines were observed in the past few hours:\n"
    headline_number = 1  # Initialize the headline number

    for news_provider in news_providers:
        print("+++++++ " + colorize_text("Checking News Provider:", 'magenta') + " " + news_provider + " " +
              colorize_text("Search Tokens: " + tokens[0] + " | " + tokens[1], 'magenta') + " in title/description")

        # Fetch the articles and store them in the cache
        provider_articles = parse_articles_from_provider(
            news_providers, news_provider, search_query, max_hours_old, max_articles, tokens)

        # Filter the cached articles to include only those that match the tokens
        matching_articles = []
        for article in provider_articles:
            title = article["title"]
            description = article["description"]
            pub_date_str = article["pub_date"]
            # or any(re.search(r'\b' + re.escape(token.lower()) + r'\b', description.lower()) for token in tokens):
            if any(re.search(r'\b' + re.escape(token.lower()) + r'\b', title.lower()) for token in tokens):

                print("++++++++ " + colorize_text("Found:",
                      'green') + " " + article["title"])
                # print("++++++++ Found: " + article["description"])
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
        print(colorize_text("Error fetching articles from {}: Status code: {}".format(
            news_provider, response.status_code), 'red'))
        return []


def parse_articles_from_response(xml_content, news_provider, max_hours_old, max_articles, tokens):
    from xml.etree import ElementTree as ET

    root = ET.fromstring(xml_content)
    articles = []

    if news_provider == "google":
        namespace = None
    elif news_provider == "bing":
        namespace = {"news": "http://www.bing.com/schema/news"}

    # Use the 'GMT' time zone for conversion
    gmt_timezone = pytz.timezone('GMT')

    for item in root.findall(".//item"):
        title = item.find("title").text
        link = item.find("link").text
        description = item.find("description").text

        if news_provider == "google":
            pub_date_str = item.find("pubDate").text
        elif news_provider == "bing":
            pub_date_str = item.find("pubDate", namespace).text
        else:
            pub_date_str = item.find("pubDate").text  # For other providers

        # Remove "GMT" from pub_date_str
        pub_date_str = pub_date_str.replace(" GMT", "")

        # Parse the date with the timezone information
        try:
            pub_date = datetime.strptime(
                pub_date_str, "%a, %d %b %Y %H:%M:%S %z")
        except ValueError:
            # If the "+0000" timezone offset is missing, try parsing without timezone information
            pub_date = datetime.strptime(pub_date_str, "%a, %d %b %Y %H:%M:%S")

        # Check if pub_date is naive (does not have timezone information)
        if pub_date.tzinfo is None:
            # Convert pub_date to GMT/UTC
            pub_date = gmt_timezone.localize(pub_date)

        if (datetime.now(pytz.utc) - pub_date) <= timedelta(hours=max_hours_old):
            if description and title:
                # Check if the article with the same title already exists
                if any(article["title"] == title for article in articles):
                    logger.warning(colorize_text(
                        f"Article with title '{title}' already exists. Skipping.", 'yellow'))
                    continue

                # print(f"++++++++{news_provider}++++++++++ " + title)
                # print(f"++++++++{news_provider}++++++++++ " + link)
                # print(f"++++++++{news_provider}++++++++++ " + description)
                # print(f"++++++++{news_provider}++++++++++ " + pub_date_str)

                article = {
                    "title": title,
                    "link": link,
                    "description": description,
                    "pub_date": pub_date_str
                }
                # prompt += f"{len(articles)}: title: {article['title']}, published: {pub_date_str}\n"
                # print("----------------- " + prompt)
                articles.append(article)
            else:
                logger.warning(colorize_text(
                    "Skipping article due to missing description or title.", 'yellow'))
                continue

            """
            if len(articles) >= max_articles:
                print("Max articles reached")
                break
            """
    return articles


def fetch_tweets(tokens, max_hours_old, bearer_token):
    prompt = "\nThe following tweets were observed in the past few hours:\n"
    tweets = []
    tweet_number = 1  # Initialize the tweet number

    search_url = "https://api.twitter.com/2/tweets/search/recent"

    additional_keywords = "crypto news"

    min_likes = 0
    min_followers = 10000

    # Get the current UTC time
    end_time = datetime.utcnow() - timedelta(seconds=10)

    # Subtract the desired number of hours to get the start time
    start_time = end_time - timedelta(hours=max_hours_old)

    # Convert to ISO 8601 format
    end_time_str = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
    start_time_str = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')

    query = f"{tokens[0]} {tokens[1]} {additional_keywords}"

    # Optional params: start_time,end_time,since_id,until_id,max_results,next_token,
    # expansions,tweet.fields,media.fields,poll.fields,place.fields,user.fields
    query_params = {
        'query': query,
        'tweet.fields': 'author_id,created_at,text,public_metrics',
        'expansions': 'author_id',
        'user.fields': 'username,name,public_metrics',
        'start_time': start_time_str,
        'end_time': end_time_str,
        'max_results': 2,
        # Add any additional fields or parameters as needed
    }

    session: Session = Session()
    headers = {
        'Authorization': f"Bearer {bearer_token}",
        'User-Agent': "v2RecentSearchPython"
    }
    session.headers.update(headers)

    print("+++++++ " + colorize_text("Checking Tweets:", 'magenta'))

    try:
        response = session.get(search_url, params=query_params)
        # This will raise an HTTPError if the HTTP request returned an unsuccessful status code
        response.raise_for_status()

    except requests.exceptions.HTTPError as e:
        if response.status_code == 429:  # Rate Limit Exceeded
            # Extract rate limit reset time from headers and wait
            reset_time = int(response.headers.get('x-rate-limit-reset', 0))
            sleep_duration = reset_time - time.time() + 5  # Adding 5 seconds as a buffer
            if sleep_duration > 0:
                logger.warning("++++ " + colorize_text(
                    f"Rate limit exceeded. Sleeping for {sleep_duration:.2f} seconds.", 'yellow'))
                time.sleep(sleep_duration)
                # Recursive call to retry after sleeping
                response = session.get(search_url, params=query_params)
            else:
                raise e
        else:
            logger.warning(
                "++++ " + colorize_text(f"Error occurred: {response.status_code} - {response.text}", 'red'))
            raise e

    json_response = response.json()
    # logger.warning(json_response)

    # Mapping user IDs to user details for easier access
    user_details = {user['id']: user for user in json_response.get('includes', {}).get(
        'users', []) if user['public_metrics']['followers_count'] >= min_followers}

    # Filter the results based on the authors with more than minimum followers, minimum likes and contain token cashtag
    cashtag = f"${tokens[0]}"
    contains_cashtag = re.compile(r'\$' + cashtag.lstrip('$'), re.IGNORECASE)
    filtered_tweets = [
        tweet for tweet in json_response.get('data', [])
        if tweet['author_id'] in user_details and tweet['public_metrics']['like_count'] >= min_likes and contains_cashtag.search(tweet['text'])
    ]

    json_response['data'] = filtered_tweets

    for tweet in json_response['data']:
        tweet_date = tweet['created_at']
        tweet_content = tweet['text']
        replies = tweet['public_metrics']['reply_count']
        retweets = tweet['public_metrics']['retweet_count']
        likes = tweet['public_metrics']['like_count']
        views = tweet['public_metrics']['impression_count']
        author_id = tweet['author_id']
        author_name = user_details[author_id]['name']
        followers_count = user_details[author_id]['public_metrics']['followers_count']

        tweetstats = f"{replies} replies, {retweets} retweets and {likes} likes"

        tweet = {
            "date": tweet_date,
            "content": tweet_content,
            "stats": tweetstats
        }

        # Filter out airdrops
        if "airdrop" not in tweet['content'].lower():
            tweets.append(tweet)
            logger.info("++++++++ " + colorize_text("Found:",
                                                    'green') + " " + tweet['content'])
            tweet_entry = f"Date: {tweet['date']}\nContent: {tweet['content']}\nStats: {tweet['stats']}\n"
            prompt += tweet_entry
            tweet_number += 1

    return tweets, prompt, tweet_number


def set_coin_dict(context="spot"):
    # Must match pair list
    spot_dict = {
        "1INCH/USDT": "1INCH crypto",
        "AAVE/USDT": "AAVE crypto",
        "ADA/USDT": "Cardano ADA crypto",
        "BTC/USDT": "Bitcoin BTC crypto",
        "ETH/USDT": "Ethereum ETH crypto DEFI",
        "DOT/USDT": "Polkadot DOT crypto",
        "LINK/USDT": "Chainlink LINK crypto DEFI",
        "DOGE/USDT": "Dogecoin DOGE crypto",
        "SOL/USDT": "Solana SOL crypto DEFI",
        "MATIC/USDT": "Polygon MATIC crypto DEFI layer2",
        "EGLD/USDT": "Elrond EGLD crypto DEFI",
        "XLM/USDT": "Stellar XLM crypto",
        "XRP/USDT": "Ripple XRP crypto",
        "XMR/USDT": "Monero XMR crypto",
        "ZEC/USDT": "Zcash ZEC crypto",
        "AVAX/USDT": "Avalanche AVAX crypto DEFI",
        "SNX/USDT": "Synthetix SNX crypto DEFI",
        "ATOM/USDT": "Cosmos ATOM crypto layer2",
        "LTC/USDT": "Litecoin LTC crypto",
        "ALGO/USDT": "Algorand ALGO crypto layer2",
        "UNI/USDT": "Uniswap UNI crypto DEFI",
        "COMP/USDT": "Compound COMP crypto DEFI",
        "YFI/USDT": "Yearn Finance YFI crypto DEFI",
        "SUSHI/USDT": "SushiSwap SUSHI crypto DEFI",
        "MKR/USDT": "Maker MKR crypto DEFI",
        # "CRV/USDT": "Curve DAO CRV crypto DEFI",
        "BAL/USDT": "Balancer BAL crypto DEFI",
        "BCH/USDT": "Bitcoin Cash BCH crypto",
        "BNB/USDT": "Binance Coin BNB crypto",
        "EOS/USDT": "EOS EOS crypto",
        "TRX/USDT": "TRON TRX crypto",
        "XTZ/USDT": "Tezos XTZ crypto DEFI",
        "NEO/USDT": "NEO NEO crypto",
        "OMG/USDT": "OMG Network OMG crypto layer2",
        "DASH/USDT": "Dash DASH crypto",
        "ZIL/USDT": "Zilliqa ZIL crypto",
        "ENJ/USDT": "Enjin Coin ENJ crypto",
        "MANA/USDT": "Decentraland MANA crypto",
        "BAT/USDT": "Basic Attention Token BAT crypto",
        "GRT/USDT": "The Graph GRT crypto DEFI",
        "FIL/USDT": "Filecoin FIL crypto",
        "KSM/USDT": "Kusama KSM crypto",
        "THETA/USDT": "Theta Network THETA crypto",
        "VET/USDT": "VeChain VET crypto",
        "ICX/USDT": "ICON ICX crypto",
        "WAVES/USDT": "Waves WAVES crypto",
        # "QTUM/USDT": "Qtum QTUM crypto",
        "SHIB/USDT": "Shiba Inu SHIB crypto",
        # "BTT/USDT": "BitTorrent BTT crypto",
        "DGB/USDT": "DigiByte DGB crypto"
    }

    futures_suffix = ":USDT"

    if context == "spot":
        coin_dict = spot_dict
    elif context == "futures":
        coin_dict = {key + futures_suffix: value for key,
                     value in spot_dict.items()}
    else:
        raise ValueError(f"Unknown context: {context}")

    return coin_dict
