WITH raw_data AS (
    SELECT * 
    FROM {{ source('kaggle', 'Steam') }}
    WHERE steam_store_available IS TRUE
),

cleaned AS (
    SELECT
        app_id,
        name,
        release_date,
        EXTRACT(YEAR FROM release_date) AS release_year,
        
        CAST(price AS FLOAT64) AS price,
        COALESCE(CAST(achievements AS INT64), 0) AS achievements_count,
        COALESCE(CAST(dlc_count AS INT64), 0) AS dlc_count,
        
        COALESCE(CAST(windows AS BOOL), FALSE) AS supports_windows,
        COALESCE(CAST(mac AS BOOL), FALSE) AS supports_mac,
        COALESCE(CAST(linux AS BOOL), FALSE) AS supports_linux,

        CASE WHEN website IS NOT NULL AND website != '' THEN 1 ELSE 0 END AS has_website,
        CASE WHEN support_url IS NOT NULL AND support_url != '' THEN 1 ELSE 0 END AS has_support_url,
        CASE WHEN support_email IS NOT NULL AND support_email != '' THEN 1 ELSE 0 END AS has_support_email,

        ARRAY_LENGTH(REGEXP_EXTRACT_ALL(screenshots, r'https://')) AS screenshot_count,
        ARRAY_LENGTH(REGEXP_EXTRACT_ALL(movies, r'https://')) AS movie_count,

        COALESCE(CAST(positive AS INT64), 0) AS positive_reviews,
        COALESCE(CAST(negative AS INT64), 0) AS negative_reviews,
        (COALESCE(CAST(positive AS INT64), 0) + COALESCE(CAST(negative AS INT64), 0)) AS total_reviews,
        
        -- YUVARLAMA EKLENDİ (Virgülden sonra 2 basamak)
        CASE 
            WHEN (COALESCE(CAST(positive AS INT64), 0) + COALESCE(CAST(negative AS INT64), 0)) = 0 THEN 0.0
            ELSE ROUND(SAFE_DIVIDE(CAST(positive AS FLOAT64), (CAST(positive AS FLOAT64) + CAST(negative AS FLOAT64))) * 100.0, 2)
        END AS positive_review_percentage,

        CASE 
            WHEN estimated_owners IN ('0 - 0', '0 .. 0') THEN 50
            WHEN estimated_owners IN ('0 - 20000', '0 .. 20,000') THEN 10000
            WHEN estimated_owners IN ('20000 - 50000', '20,000 .. 50,000') THEN 35000
            WHEN estimated_owners IN ('50000 - 100000', '50,000 .. 100,000') THEN 75000
            WHEN estimated_owners IN ('100000 - 200000', '100,000 .. 200,000') THEN 150000
            WHEN estimated_owners IN ('200000 - 500000', '200,000 .. 500,000') THEN 350000
            WHEN estimated_owners IN ('500000 - 1000000', '500,000 .. 1,000,000') THEN 750000
            WHEN estimated_owners IN ('1000000 - 2000000', '1,000,000 .. 2,000,000') THEN 1500000
            WHEN estimated_owners IN ('2000000 - 5000000', '2,000,000 .. 5,000,000') THEN 3500000
            WHEN estimated_owners IN ('5000000 - 10000000', '5,000,000 .. 10,000,000') THEN 7500000
            WHEN estimated_owners IN ('10000000 - 20000000', '10,000,000 .. 20,000,000') THEN 15000000
            WHEN estimated_owners IN ('20000000 - 50000000', '20,000,000 .. 50,000,000') THEN 35000000
            WHEN estimated_owners IN ('50000000 - 100000000', '50,000,000 .. 100,000,000') THEN 75000000
            WHEN estimated_owners IN ('100000000 - 200000000', '100,000,000 .. 200,000,000') THEN 150000000
            ELSE 10000
        END AS estimated_owners_avg,

        LOWER(COALESCE(genres, '')) AS genres,
        LOWER(COALESCE(categories, '')) AS categories,
        LOWER(COALESCE(tags, '')) AS tags,
        LOWER(COALESCE(supported_languages, '')) AS supported_languages,
        LOWER(COALESCE(full_audio_languages, '')) AS full_audio_languages -- DİL VE SESLENDİRME EKLENDİ

    FROM raw_data
)

SELECT * FROM cleaned