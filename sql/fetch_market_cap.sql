SELECT 
    gmbp.company_id as company_id,
    gmbp.day_date as day_date,
    gmbp.market_cap_usd as market_cap_usd
FROM BIOTECH_PROJECT.READ_ONLY.MONTHLY_EQUITY_DATA as gmbp
ORDER BY gmbp.COMPANY_ID, gmbp.PRICING_DATE;