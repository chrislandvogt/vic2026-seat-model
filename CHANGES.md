# Release history

Backtest scores are mean winner Brier / log loss over 2006-2022 unless stated (lower is better).

**v1.25 (22 November 2026): high-profile independents.** New independents endorsed by Climate 200 (2026: Hawthorn,
Mornington, Footscray; Kew's Sophie Torney is recontesting and starts from her own vote) start higher, by half the
2022 gap: +0.62 on the log of the independent's median share (the four 2022 Climate 200 candidates averaged 3.5 times the
model's generic new-independent vote). The extra votes come from Labor, the Greens and minor parties, not the Liberals,
as in 2022 (Labor 40%, Greens 30%, others 29% of the teal vote; the Liberal vote held). 2022 backtest, six seeds, boost
for each seat estimated from the other three: Brier 0.1074 -> 0.1046, log loss 0.1812 -> 0.1788, first-preference error
3.60 -> 3.54 points. One election and four seats; the choice of where the votes come from was informed by those seats.
A first version that took the votes from every party in proportion made 2022 worse (Brier 0.1132). Effect on 2026:
Footscray Labor 76% -> 62%, independent 5% -> 20%; statewide unchanged.

**v1.24 (22 November 2026): One Nation and Greens preference flows.** (1) Coalition v One Nation final pairs: Labor
ballots 0.69 and Greens ballots 0.65 to the Coalition (South Australia, March 2026; previously 0.80 and 0.92). In 16
out-of-state Coalition-v-One Nation final counts Labor and Greens preferences gave the Coalition 0.52-0.77 (pooled 0.64),
and the model's count understated One Nation by 3.4 points. (2) Minor-party and independent preferences between Labor and
the Greens: others 0.39 to Labor at Labor-v-Greens finals, and minor-party/independent ballots reaching the Greens more
often mid-count. On 72 real counts with the Greens in the final two the model had the Greens 1.6-1.8 points too low
(-0.6 after), final pair right 58 -> 63 of 72. Victorian backtests unchanged (0.1197 -> 0.1198; One Nation barely stood).
Effect on 2026: One Nation 7.6 -> 8.6 seats, Greens 4.7 -> 5.4, Coalition 52.5 -> 51.3; Northcote to the Greens.

**v1.23 (18 November 2026): Greens in young seats.** The Greens' seat vote moves toward seats with more residents aged
20-34 (2021 Census) and away from older seats, 0.49 points per standard deviation, statewide total unchanged. The
Greens beat the model in young seats in four of the five elections. 0.1228 -> 0.1197, log loss 0.2091 -> 0.2040. The
coefficient is leave-one-election-out in the backtests. About 13 inner-city contests drive the gain.

**v1.22 (18 November 2026): four structural changes.** Labor and Coalition swings uniform in points (floor 40% of the
previous seat vote); seat noise 1.35 times larger in country seats and 0.85 in Melbourne; a new independent's median
vote rises with the seat's margin and where the member retires; in retiring members' seats the Labor-v-Coalition split
moves by 0.18 of the retiring party's state-minus-federal vote gap. 0.1254 -> 0.1228, log loss 0.2169 -> 0.2091.

**v1.21 (17 November 2026): poll trend.** A kernel-weighted straight line through the polls (35-day kernel) instead of a
kernel-weighted mean, which lagged a moving trend. 0.1275 -> 0.1255.

**v1.20 (17 November 2026): candidates and sitting members.** Returning independents start from their own vote
(median retention 0.78); new independents lognormal about 2.2%; first-term members gain 3.0 points; the federal relative
swing coefficient falls from 0.5 to 0.2; the number of minor candidates on the ballot moves the other vote.
0.1468 -> 0.1275.

**v1.19 (15 November 2026): seat swing spread** measured over all five election pairs 2002-2022 instead of one.
