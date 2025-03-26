# Memo: Who will buy next week, and should we send them a coupon?

**To:** E-commerce leadership  **From:** Analytics  **Date:** 2026-11-01

## Recommendation

**Roll out the targeted 10% coupon** to the customers our
model ranks in the top 10% most likely to buy. In a controlled
test it raised the share of customers who bought within a week from
2.4% to 2.5%
(+5%) without a meaningful loss of revenue per customer.

## What we found

1. **A few customers carry the business.** The top 1% of visitors produce
   51% of revenue, and only 13%
   of visitors buy anything in two months.
2. **The biggest leak is before the cart.** 20% of visitors
   ever add something to their cart; once they do, 54% buy.
   Improving product pages and prices matters more than the checkout.
3. **Promotions work on traffic.** The mid-November sale brought
   2.6x the usual daily visitors and lifted conversion from
   5.9% to 7.6%.
4. **Weekends convert better** and cheaper products sell more readily; both are statistically
   solid, not noise.
5. **Electronics is the core:** 76% of revenue.

## The prediction model, in plain terms

Every week the model gives each recently active customer a probability of buying in the next
seven days. If we contact the top 10% it flags, we reach
50% of next week's buyers - about
5.0 times better than picking customers at random.
The strongest signals are recent activity (visits and product views in the last week), how long
ago the customer last visited, and whether past carts turned into orders.

## What it would take to act

- Score customers weekly (automated; results are available per customer through an internal
  service and a dashboard).
- Run any coupon or campaign as a proper A/B test, fixed in size and length in advance.
  Checking results every day and stopping early would make roughly
16% of "wins" false alarms.
- Next step: target customers whose behaviour a coupon actually *changes* (an "uplift" model),
  not just those already likely to buy, so discounts are not given to people who would have
  bought anyway.
