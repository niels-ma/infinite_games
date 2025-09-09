<div align="center">

<img src="docs/infinite-games.jpeg" alt="Project Logo" width="200"/>

# **Infinite Games** 


[Discord](https://discord.gg/qKPeYPc3) • [Dashboard](https://app.hex.tech/1644b22a-abe5-4113-9d5f-3ad05e4a8de7/app/5f1e0e62-6072-4440-9646-6d2b60cd1674/latest) •
[Website](https://www.infinitegam.es/) • [Twitter](https://twitter.com/Playinfgames) •  [Network](https://taostats.io/) 

---

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

</div>



<!-- update with discord invite link -->

</div>

  <!--
  <a href="">Twitter</a> -->
    
  <!-- <a href="">Bittensor</a> -->


# Forecasting of Future Events

##  Introduction


We incentivize the prediction of future events. The prediction space is based on binary future events such as the ones listed on [Polymarket](https://polymarket.com/). We are always actively expanding to new data providers. Most of this data is then processed by an LLM pipeline which handles the event generation. We are focused on *judgemental forecasting* rather than *statistical forecasting*. We hence expect the models used by miners to be LLMs. 

### High-level mechanism

Miners submit their predictions to validators. Each prediction has to be done early enough before the event underlying the prediction settles. Once the event settles, the validators that received the prediction score the miner.

## LLMs open new predictive capabilities

Making predictions is a hard task that requires cross-domain knowledge and intuition. It is often limited in explanatory reasoning and domain-specific (the expert in predicting election results will differ from the one predicting the progress in rocket-engine technology) ([1]). At the same time it is fundamental to human society, from geopolitics to economics. 

LLMs approach or surpass human forecasting abilities. They near on average the crowd prediction on prediction market events ([1]), and surpass humans in predicting neuroscience results ([2]). They are also shown to be calibrated with their predictions i.e confident when right. Through their generalization capabilities and unbounded information processing, LLMs have the potential to automate the prediction process or complement humans. 


### Real-world applications

The value of the subnet first relies in the improvement of the efficiency of prediction markets. This value can be extracted by validators through arbitrage. The validators may obtain a better knowledge of the probability of an event settling and communicate this information to a prediction market by opening a position. 

The first applications built on top of our subnet could be related to prediction markets. A trader could query our market to obtain the most up to date and relevant predictions to their portfolio based on the current news landscape (LLMs would be constantly aggregating the most up to date and relevant news articles). They could then readjust their positions accordingly or trade directly on this information. 

In the long term, a validator could provide paid economic forecasts or more generally the output of any forward-looking task addressed to an LLM ([2]). A customer might then provide a series of paid sub-queries related to the information they aim at retrieving.


## Miners 

Miners compete by sending to the validators for each binary event $E$ their estimation of the probability $p$ that $E$ is realized. For example, $E$ could be *o3 is released to the public by January 15th 2025*.

The prediction $p$ should be a **float number between 0.0 and 1.0**; e.g. 0.75 = 75% probability. Any prediction outside of these bounds will be clipped to the interval $(0,1)$.


### Miner strategy 

A reference providing a **baseline miner** strategy is the article ["Approaching Human Level Forecasting with Langage Models"](https://arxiv.org/html/2402.18563v1?s=35) ([1]). The authors fine-tune an LLM to generate predictions on binary events (including the ones listed on Polymarket) which nears the performance of human forecasters when submitting a forecast for each prediction, and which beats human forecasters in a setting where the LLM can choose to give a prediction or not based on its confidence.

According to the article, the performance of forecasting LLMs depends significantly on the amount of data one can retrieve for a given prediction or event (for example prediction market data). If our subnet is able to continually produce new synthetic data, miners should be able to beat the SoA.


## Validators

Validators record the miners' predictions and score them once the events settle. At each event settlement, a score is added to the moving average of the miner's score. We implement a **cutoff** for the submission time of a prediction. The cutoff is set at 24 hours before the resolution date for most events.

## Scoring Rule

### Brier Score (Legacy)

For a binary event $E_q$, a miner $i$ submits a prediction $p_i$ representing the probability that the event will occur. Let the outcome $o_q$ be defined as:
- $o_q = 1$ if the event is realized,
- $o_q = 0$ otherwise.

The Brier score $S(p_i, o_q)$ for the prediction is given by:
- **If $o_q = 1$:**  
  
  $$S(p_i, 1) = 1 - (1 - p_i)^2$$
  
- **If $o_q = 0$:**  
  $$S(p_i, 0) = 1 - p_i^2.$$

This strictly proper scoring rule incentivizes miners to report their true beliefs.

#### Time Series of Predictions

For each event, the forecast period between the issue date and the resolution date is divided into $n$ submission intervals. During each interval, miners must submit a prediction. The validator records these predictions as a **time series**. If a miner fails to submit a prediction in any interval, an "empty" prediction is registered, which is assigned the worst possible score according to the Brier rule.

Once the event resolves, the validator computes the Brier score for each prediction in the time series, producing a corresponding series of scores. These scores are then aggregated into a **weighted average**, where later (more recent) predictions are weighted more heavily. Specifically, for submission interval $k$ (with $k = 0, 1, \dots, n-1$), the weight is given by:

$$w_k = \exp\\left(-\frac{n}{n-k} + 1\right)$$.

A detailed explanation of this process is available [here](docs/peer-scoring.md).


### Peer Scoring

In the updated system, the **peer scoring** mechanism replaces the legacy Brier score method. For each binary event, a miner $i$ submits a prediction $p_i$ representing their belief that the event will occur.

For miner \(i\), the peer score on a given event that resolves positively ($o_q=1$) is defined as:

$$S(p_i, 1) = \frac{1}{n}\sum_{j \neq i} \Bigl(\log(p_i) - \log(p_j)\Bigr)
= \log(p_i) - \frac{1}{n}\sum_{j \neq i}\log(p_j)$$.

This calculation is performed for every submission interval, and the resulting peer scores are stored as a time series. Exacty as above, in case of no prediction being submitted a miner gets the worst possible score under the peer scoring rule. The same exponential weighting scheme is applied to compute a weighted average peer score for each miner, with the weight for interval $k$ (where $k = 0, 1, \dots, n-1$) defined as:

$$w_k = \exp\\left(-\frac{n}{n-k} + 1\right)$$.

For each event, the weighted average peer score is then added to a moving average $M_i$ calculated over the last $N$ events — where $N$ is proportional to the number of events generated and resolved during an immunity period.

Finally, the validator applies an extremising function to $M_i$:
$$F(M_i) = \max(M_i, 0)^2$$
and normalizes the resulting scores across all miners.


## Roadmap
- [x] Scoring with exponentially decreasing weights until settlement date and linear differentiation mechanism 
- [x] Synthetic event generation with central resolution using ACLED data 
- [x] Scoring with exponential differentiation mechanism 
- [x] Comprehensive and granular analytics 
- [x] Synthetic event generation with UMA resolution - human verifiers resolve our events through the OOv2 
- [x] Synthetic event generation from news data using an LLM 

- [x] Validator v2 - modular and much higher throughput 
- [x] Scoring v2 (batches, peer score)
- [x] Exposing the silicon crowd predictions 

- [ ] Decentralisation of event generation and validator dynamic desirability (inspired from SN13)
- [ ] Trustless event resolution
- [ ] Commit-reveal on the miners' predictions
- [ ] Scoring a reasoning component
- [ ] Data generation for iterative fine-tuning of prediction focused LLMs
- [ ] Reasoning component
- [ ] Data generation for iterative fine-tuning of forecasting LLMs



## Running a miner or validator

Regarding instructions and requirements, see [here](docs/validator.md) for validators and [here](docs/miner.md) for miners.

## Setting up a bittensor wallet
A detailed explanation of how to set up a wallet can be found [here](https://docs.bittensor.com/getting-started/wallets). 
We also provide some indications [here](docs/wallet-setup.md).


## References

| Reference ID | Author(s) | Year | Title |
|--------------|-----------|------|-------|
| 1 | Halawi and al. | 2024| [Approaching Human Level Forecasting with Langage Models](https://arxiv.org/html/2402.18563v1?s=35)|
| 2 | Luo and al. | 2024 | [LLM surpass human experts in predicting neuroscience results](https://arxiv.org/pdf/2403.03230.pdf)|


---

## License
This repository is licensed under the MIT License.
```text
# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
```
