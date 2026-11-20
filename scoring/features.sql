WITH bureau_history AS (
    SELECT
        SK_ID_CURR,
        COUNT(*) AS bureau_credit_count,
        COUNT(*) FILTER (WHERE CREDIT_ACTIVE = 'Active') AS bureau_active_count,
        COUNT(*) FILTER (WHERE DAYS_CREDIT >= -365) AS bureau_credits_1y,
        MAX(DAYS_CREDIT) AS bureau_last_credit_day,
        MAX(CREDIT_DAY_OVERDUE) AS bureau_max_overdue_days,
        SUM(AMT_CREDIT_SUM_DEBT) AS bureau_total_debt,
        SUM(AMT_CREDIT_SUM_OVERDUE) AS bureau_total_overdue
    FROM read_csv($bureau, header = true)
    WHERE DAYS_CREDIT < 0 AND DAYS_CREDIT_UPDATE < 0
    GROUP BY SK_ID_CURR
), previous_history AS (
    SELECT
        SK_ID_CURR,
        COUNT(*) AS prev_application_count,
        AVG((NAME_CONTRACT_STATUS = 'Refused')::DOUBLE) AS prev_refusal_rate,
        AVG((NAME_CONTRACT_STATUS = 'Approved')::DOUBLE) AS prev_approval_rate,
        MAX(DAYS_DECISION) AS prev_last_decision_day,
        AVG(AMT_CREDIT) FILTER (WHERE NAME_CONTRACT_STATUS = 'Approved') AS prev_avg_approved_credit
    FROM read_csv($previous, header = true)
    WHERE DAYS_DECISION < 0
    GROUP BY SK_ID_CURR
), payment_history AS (
    SELECT
        SK_ID_CURR,
        COUNT(*) AS past_payment_count,
        AVG((DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT)::DOUBLE) AS past_late_payment_rate,
        MAX(GREATEST(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT, 0)) AS past_max_late_days
    FROM read_csv($payments, header = true)
    WHERE DAYS_INSTALMENT < 0 AND DAYS_ENTRY_PAYMENT < 0
    GROUP BY SK_ID_CURR
)
SELECT
    a.SK_ID_CURR,
    a.TARGET,
    a.EXT_SOURCE_1,
    a.EXT_SOURCE_2,
    a.EXT_SOURCE_3,
    a.AMT_INCOME_TOTAL,
    a.AMT_CREDIT,
    a.AMT_ANNUITY,
    a.AMT_GOODS_PRICE,
    a.AMT_CREDIT / NULLIF(a.AMT_INCOME_TOTAL, 0) AS credit_to_income,
    a.AMT_ANNUITY / NULLIF(a.AMT_INCOME_TOTAL, 0) AS annuity_to_income,
    a.AMT_ANNUITY / NULLIF(a.AMT_CREDIT, 0) AS annuity_to_credit,
    a.AMT_CREDIT / NULLIF(a.AMT_GOODS_PRICE, 0) AS credit_to_goods_price,
    -a.DAYS_BIRTH / 365.25 AS age_years,
    -NULLIF(a.DAYS_EMPLOYED, 365243) / 365.25 AS employment_years,
    a.DAYS_ID_PUBLISH,
    a.DAYS_REGISTRATION,
    a.DAYS_LAST_PHONE_CHANGE,
    a.CNT_CHILDREN,
    a.CNT_FAM_MEMBERS,
    a.OWN_CAR_AGE,
    a.REGION_RATING_CLIENT,
    a.REGION_POPULATION_RELATIVE,
    a.AMT_REQ_CREDIT_BUREAU_YEAR,
    a.DEF_30_CNT_SOCIAL_CIRCLE,
    a.NAME_CONTRACT_TYPE,
    a.NAME_INCOME_TYPE,
    a.NAME_EDUCATION_TYPE,
    a.NAME_HOUSING_TYPE,
    a.OCCUPATION_TYPE,
    COALESCE(b.bureau_credit_count, 0) AS bureau_credit_count,
    COALESCE(b.bureau_active_count, 0) AS bureau_active_count,
    COALESCE(b.bureau_credits_1y, 0) AS bureau_credits_1y,
    b.bureau_last_credit_day,
    b.bureau_max_overdue_days,
    b.bureau_total_debt,
    b.bureau_total_overdue,
    COALESCE(p.prev_application_count, 0) AS prev_application_count,
    p.prev_refusal_rate,
    p.prev_approval_rate,
    p.prev_last_decision_day,
    p.prev_avg_approved_credit,
    COALESCE(i.past_payment_count, 0) AS past_payment_count,
    i.past_late_payment_rate,
    i.past_max_late_days
FROM read_csv($applications, header = true) AS a
LEFT JOIN bureau_history AS b USING (SK_ID_CURR)
LEFT JOIN previous_history AS p USING (SK_ID_CURR)
LEFT JOIN payment_history AS i USING (SK_ID_CURR)
ORDER BY a.SK_ID_CURR;
