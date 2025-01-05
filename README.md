# Freqtrade

Freqtrade is a free and open source crypto trading bot written in Python.

This repository is a collection of strategies, configurations, dry-run and backtest results I collected overtime.

### Disclaimer

I am not a financial advisor and I am not responsible for any financial loss you might incur using this repository.

### Quickstart

This will install freqtrade and dependencies to work with this repository tools. Or you can just ignore this and look through the repository.

```bash
# Clone the repository
git clone https://github.com/TheoBrigitte/freqtrade.git
cd freqtrade
scripts/setup.sh
```

#### Update freqtrade

Freqtrade can later be updated with the following command

```bash
scripts/setup.sh -u
```

### Repository structure

- `backtest_results/` - Contains my backtest results, good or bad, they are all here
- `config/`          - Contains a bunch of configurations I found, some I used, some I didn't
- `scripts/`          - Contains some scripts I use to work with freqtrade and this repository
- `pairlist/`         - Contains a collection of pairlists
- `strategies/`       - Contains all strategies I found and played with. Folder names are arbitrary and each folder might contain additional configurations for the specific strategy(ies)
- `sources/`          - Contains sources of strategies I found. They mainly are git submodules to other repositories

### How I use this repository

Freqtrade virtual environment should be loaded first

```bash
source freqtrade/.venv/bin/activate
```

### Update pairlist and download data

```bash
scripts/fw generate-static-pairlist ./pairlist/pairlist-volume-usdt.json
mv pairlist.json pairlist/new_pairlist.json
ln -fsrv pairlist/new_pairlist.json pairlist/pairlist_futures.json
scripts/fw download-data --timeframes 1m 3m 5m 15m 30m 1h 4h 12h 1d --timerange 20241201-20250101
```

### Run tests

I usually run lookahead-analysis, backtesting and plot profit in order to see how a strategy performs.

```bash
fw lookahead-analysis strategies/BinHV45/BinHV45.py --timerange 20240101-20250101
fw backtesting strategies/BinHV45/ --timerange 20240101-20250101
fw plot-profit backtest_results/backtest-result-2025-01-01_01-01-01.json
```

Based on my observations I might tweak the strategy or the configuration and run the tests again.
Configuration tweaking for this strategy would go in `strategies/BinHV45/config.json`.

### Backtest results interpretation

Many aspects comes into play when it comes to evaluating a strategy. Backtest results are the first step I use to evaluate a strategy.
Once I found something that looks promising I will run a dry-run over couple of weeks and ideally months to see how it performs in near real conditions.

In a nutshell those are the most important metrics I look at:

Expectancy: The average amount you can expect to win (or lose) per trade with this strategy. Above 0.2 is usually considered good.
Profit factor: The ratio of gross profit to gross loss. Above 2 is usually considered good.
Avg profit: The average profit per trade. Above 1% is usually considered good.
Drawdown: The maximum drawdown of the strategy. This depends on your risk tolerance, but below 10% is usually considered good.

This is a very high level overview and there are many other metrics and aspects to consider.

Here is a good knowledge base to learn from: https://botacademy.ddns.net

### Navigate backtest results

I usually use the following command to navigate through backtest results

```bash
scripts/backtest-browser.sh
```

# Credits

- https://github.com/freqtrade/freqtrade
