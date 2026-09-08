WITH raw_data AS (
    SELECT * 
    FROM {{ source('kaggle', 'Steam') }}
    WHERE steam_store_available IS TRUE
      -- Ücretsiz ve F2P oyunları eliyoruz
      AND CAST(price AS FLOAT64) > 0
      AND (genres IS NULL OR NOT REGEXP_CONTAINS(LOWER(genres), r'free to play'))
      -- Null / Eksik değerleri temizleme
      AND price IS NOT NULL
      AND release_date IS NOT NULL
      AND estimated_owners IS NOT NULL
      AND TRIM(estimated_owners) != ''
      -- 0 Oyuncusu olan ölü kayıtları eliyoruz
      AND TRIM(estimated_owners) NOT IN ('0 - 0', '0 .. 0')
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

        COALESCE(ARRAY_LENGTH(REGEXP_EXTRACT_ALL(screenshots, r'https://')), 0) AS screenshot_count,
        COALESCE(ARRAY_LENGTH(REGEXP_EXTRACT_ALL(movies, r'https://')), 0) AS movie_count,

        COALESCE(CAST(positive AS INT64), 0) AS positive_reviews,
        COALESCE(CAST(negative AS INT64), 0) AS negative_reviews,
        (COALESCE(CAST(positive AS INT64), 0) + COALESCE(CAST(negative AS INT64), 0)) AS total_reviews,
        
        CASE 
            WHEN (COALESCE(CAST(positive AS INT64), 0) + COALESCE(CAST(negative AS INT64), 0)) = 0 THEN 0.0
            ELSE ROUND(SAFE_DIVIDE(CAST(positive AS FLOAT64), (CAST(positive AS FLOAT64) + CAST(negative AS FLOAT64))) * 100.0, 2)
        END AS positive_review_percentage,

        TRIM(estimated_owners) AS estimated_owners_raw,

        LOWER(COALESCE(genres, '')) AS genres,
        LOWER(COALESCE(categories, '')) AS categories,
        LOWER(COALESCE(tags, '')) AS tags,
        LOWER(COALESCE(supported_languages, '')) AS supported_languages,
        LOWER(COALESCE(full_audio_languages, '')) AS full_audio_languages

    FROM raw_data
)

SELECT * FROM cleaned