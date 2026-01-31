# APM466 Assignment 1 - Canadian Government Bond Data

## Overview

This dataset contains historical closing prices for **42 Canadian Government Bonds** from the Frankfurt Exchange, collected from Business Insider Markets for the period **January 5-16, 2026** (10 weekdays).

## Data Files

| File | Description |
|------|-------------|
| `bonds_final_price_matrix.csv` | Complete dataset with all bond metadata and prices |
| `bonds_clean_matrix.csv` | Clean format with renamed columns for easy use |
| `bonds_final.json` | JSON format of the complete dataset |

## Data Structure

### Column Definitions

| Column | Description |
|--------|-------------|
| ISIN | International Securities Identification Number |
| Coupon | Coupon rate as percentage string (e.g., "4.0000%") |
| Coupon_Rate | Coupon rate as decimal number (e.g., 4.0) |
| Maturity_Date | Bond maturity date (YYYY-MM-DD format) |
| Issue_Date | Bond issue date (YYYY-MM-DD format) |
| Years_to_Maturity | Years to maturity from January 5, 2026 |
| Jan05-Jan16 | Daily closing prices for each weekday |

### Price Dates

The 10 weekdays covered are:
- Week 1: Jan 5, 6, 7, 8, 9 (Mon-Fri)
- Week 2: Jan 12, 13, 14, 15, 16 (Mon-Fri)

## Bond Summary

- **Total Bonds**: 42
- **Maturity Range**: Feb 2026 to Dec 2035 (~0.07 to ~9.9 years)
- **Coupon Range**: 0.25% to 8.00%
- **Currency**: CAD
- **Rating**: Aaa (Moody's)
- **Issuer**: Government of Canada

## Data Quality

### Complete Data (38 bonds)
Most bonds have complete or interpolated price data for all 10 days.

### Sparse Data (4 bonds - may need to exclude)
The following bonds have very limited price data (only 1-3 original data points):

| ISIN | Coupon | Maturity | Notes |
|------|--------|----------|-------|
| CA135087E679 | 1.50% | 2026-06-01 | Only 1 data point |
| CA135087F825 | 1.00% | 2027-06-01 | Only 1 data point |
| CA135087J397 | 2.25% | 2029-06-01 | Only 1 data point |
| CA135087VW17 | 8.00% | 2027-06-01 | Only 1 data point |

**Recommendation**: Consider excluding these 4 bonds from yield curve analysis due to insufficient price data, leaving 38 bonds with reliable data.

## Usage Notes

1. **Yield Curve Construction**: Use bonds sorted by maturity date for bootstrapping spot rates
2. **Semi-annual Coupons**: All bonds pay coupons semi-annually
3. **Interpolation**: Missing prices were linearly interpolated where possible
4. **Day Count**: Use actual/365 or actual/actual for Canadian government bonds

## Source

Data collected from:
- Short-term bonds: https://markets.businessinsider.com/bonds/finder?borrower=71&maturity=shortterm&bondtype=2,3,4,16&currency=184&country=19
- Mid-term bonds: https://markets.businessinsider.com/bonds/finder?borrower=71&maturity=midterm&bondtype=2,3,4,16&currency=184&country=19

## Files in Package

```
APM466_Bond_Data/
├── bonds_final_price_matrix.csv    # Full dataset
├── bonds_clean_matrix.csv          # Clean format
├── bonds_final.json                # JSON format
└── APM466_Bond_Data_README.md      # This file
```
