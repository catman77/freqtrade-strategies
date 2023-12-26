
Run TM3live strategy

```sh
freqtrade trade \
	--config user_data/configs/TM3Live.dev.json \
	--freqaimodel CatboostFeatureSelectedRegressorV1 \
	--strategy TM3Live \
	--logfile user_data/logs/TM3Live.dev.log \
	--db-url sqlite:///user_data/db/TM3Live.dev.sqlite
```



## TM3MultiClass

run TM3MultiClass strategy

```sh
freqtrade trade \
	--config user_data/configs/TM3MultiClass.dev.json \
	--freqaimodel CatboostFeatureSelectedMultiTargetClassifierV1 \
	--strategy TM3MultiClass \
	--logfile user_data/logs/TM3MultiClass.dev.log \
	--db-url sqlite:///user_data/db/TM3MultiClass.dev.sqlite
```


## TM3BinaryClass

run TM3BinaryClass strategy

```sh
freqtrade trade \
	--config user_data/configs/TM3BinaryClass.dev.json \
	--freqaimodel CatboostFeatureSelectMultiTargetBinaryClassifierV1 \
	--strategy TM3BinaryClass \
	--logfile user_data/logs/TM3BinaryClass.dev.log \
	--db-url sqlite:///user_data/db/TM3BinaryClass.dev.sqlite
```

```sh
freqtrade trade \
	--config user_data/configs/TM3BinaryClass.dev.json \
	--freqaimodel CatboostFeatureSelectMultiTargetBinaryClassifierV2 \
	--strategy TM3BinaryClass \
	--logfile user_data/logs/TM3BinaryClass.dev.log \
	--db-url sqlite:///user_data/db/TM3BinaryClass.dev.sqlite
```


```sh
freqtrade backtesting \
	--timerange "20231101-20231221" \
	--config user_data/configs/TM3BinaryClass.bt.json \
	--freqaimodel CatboostFeatureSelectMultiTargetBinaryClassifierV2 \
	--strategy TM3BinaryClass
```


freqtrade download-data --timerange "20231101-" -t 1h 4h 12h 1d -p ADA/USDT:USDT BTC/USDT:USDT --exchange binance
freqtrade  download-data --exchange binance --pairs BTC/USDT:USDT ADA/USDT:USDT --trading-mode futures --timerange 20230101- -t 1h 4h 6h 12h 1d