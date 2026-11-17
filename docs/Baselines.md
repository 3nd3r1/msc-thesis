# Baselines

What the published semantic filter methods measure, on which data, and whether any of it has links between rows.
Read on 2026-11-17 from the papers, nothing run.

## Overview

| method | query | guarantee | cost metric | quality metric | models | code |
| --- | --- | --- | --- | --- | --- | --- |
| SUPG, Kang et al. VLDB 2020, arXiv 2004.00827 | recall or precision target with a fixed oracle budget | P(recall ≥ T) ≥ 1 - δ, asymptotic (CLT) | budget fixed, 1,000 or 10,000 oracle calls | precision of the returned set at the recall target | DNN proxies, human or Mask R-CNN oracle | github.com/stanford-futuredata/supg, with datasets |
| BARGAIN, Zeighami et al. 2025, arXiv 2509.02896 | accuracy, precision or recall target | P(target met) ≥ 1 - δ, finite sample (betting confidence sequences) | % of oracle calls avoided (accuracy target), budget k = 400 (precision and recall targets) | precision at T = 0.9 recall, δ = 0.1, mean of 50 runs | gpt-4o-mini proxy, gpt-4o oracle | not stated in the paper |
| LOTUS sem_filter, Patel et al. 2024, arXiv 2407.11418 | recall and precision targets together | SUPG style, δ | oracle calls against the target | observed recall and precision against the 70B | Llama 3.1 8B proxy, 70B oracle | github.com/lotus-data/lotus |
| PROBE, Zhao et al. Nov 2026, arXiv 2611.01454 | quality and cost tradeoff α, sampling budget B = 10% | none, F1 estimated with a calibration sample | delegation rate and speedup | F1 against human labels | Gemma 4 26B oracle, Qwen3-VL-Reranker-2B scores | github.com/DamonZhao-sfu/PROBE |
| CSV, Hou et al. VLDB 2026, arXiv 2603.04799 | label every row by cluster voting | error bound per cluster | number of LLM calls | accuracy and F1 | Llama 3.1 8B as the reference LLM | github.com/Anto-an/CSV_SemanticFilter |
| CascadeFilter, He et al. Nov 2026, arXiv 2611.08367 | accuracy target relative to the oracle | finite population betting test | token cost in dollars | accuracy and macro F1 against benchmark labels | Qwen3-4B small model, GPT-5.5 oracle | only a three record example |

## Datasets

SUPG uses ImageNet (hummingbirds, 0.1% positive, human labels), night-street (cars in video frames, resampled to 4% positive, Mask R-CNN labels), OntoNotes (city relation, 2.5%, human) and TACRED (employee relation, 2.4%, human).

BARGAIN uses the four SUPG datasets plus four LLM tasks of 1,000 random rows each, labelled by gpt-4o.
Reviews, 855 Steam reviews, whether the review talks about another game more positively, 23% positive.
Court, CourtListener Supreme Court opinions, whether the opinion reverses the lower court, 59%.
Screenplay, movie scripts, whether the protagonist decides on false information, 22%.
Wiki, ConvoKit Wikipedia talk page discussions, whether the discussion led to a revert, 25%.

LOTUS evaluates the filter on FEVER fact checking.

PROBE uses the 14 lowest positive rate queries of SemBench and LRO-Bench, filters and joins over text and images, 266 to 525,625 candidates, positive rates 2e-5 to 0.14.

CSV and CascadeFilter use IMDB reviews (50,000), Codebase (9,378 users from BIRD `codebase_community`, the `AboutMe` text, gpt-4o labels), AirDialogue (402,035 dialogues), Twitter hate speech (24,783 tweets) and FEVER.
CascadeFilter adds 50 Yelp aspect queries.

## Headline numbers

BARGAIN at T = 0.9 against SUPG, precision at the recall target: Reviews 39.7 vs 34.3, Court 82.0 vs 77.5, Screenplay 33.0 vs 27.9, Wiki 50.9 vs 45.9.
Under the accuracy target BARGAIN avoids 37 to 59% of oracle calls on Reviews, Court and Wiki and at most 11% on Screenplay, where SUPG avoids 1.5 to 24%.

PROBE reaches a geometric mean F1 of 0.762 at 12.8% delegation, against 0.797 for calling the LLM on every row.
In PROBE's runs BARGAIN reaches 0.473 and LOTUS 0.485, because accuracy targets are met by the negatives on rare predicates.

CSV reports 1.28 to 355 times fewer LLM calls than LOTUS and BARGAIN at similar accuracy.

## Links between rows

None of these evaluations uses links between rows.

- Codebase in CSV and CascadeFilter is the same BIRD database as our Stack data, so users can be linked by who answers or comments on whom. Its predicates (a social media link, interest in computer science, factual identification) read the `AboutMe` text, so links may not help, as on Stack.
- night-street is video, where consecutive frames share labels, but the oracle is a detector and the positives were resampled.
- Court, Reviews and Wiki have natural links: citations between opinions, the same game or author, and the same page or editors. But each is a random sample of 1,000 rows, so few links survive.
- IMDB reviews can be grouped by movie, and FEVER claims by the Wikipedia page of their evidence. Neither is checked.
- SemBench and LRO-Bench have no real row links, see the 2026-11-16 log.

## What a comparison would look like

The closest metric to ours is BARGAIN's recall target query: a 0.9 recall target, δ = 0.1, a budget of 400 calls, and the precision of the returned set, mean over 50 runs.
In BARGAIN, SUPG and LOTUS the oracle is the ground truth, while PROBE and CascadeFilter report against human labels.
None of them sends the oracle to rows through links, so the place to add links is which rows get sampled and verified, as PROBE does with a kNN graph and a separate calibration sample.

Open:

- whether BARGAIN's code is public
- whether Court or Wiki can be rebuilt with their links from CourtListener and ConvoKit at full size
- whether the Codebase labels from CSV are released
