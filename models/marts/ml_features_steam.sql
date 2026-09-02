WITH stg AS (
    SELECT * FROM {{ ref('stg_steam_games') }}
)

SELECT
    app_id,
    name,
    
    -- Target Variables (Tahmin Edilecek Sütunlar)
    price,                           -- Fiyat Tahmini Regresyonu
    positive_review_percentage,      -- Review Skor Tahmini (%)
    estimated_owners_avg,            -- Oyuncu Sayısı Tahmini

    -- Temel Özellikler (Numeric Features)
    release_year,
    achievements_count,
    dlc_count,
    COALESCE(screenshot_count, 0) AS screenshot_count,
    COALESCE(movie_count, 0) AS movie_count,
    has_website,
    has_support_url,
    has_support_email,
    
    -- Platformlar (Binary)
    IF(supports_windows, 1, 0) AS supports_windows,
    IF(supports_mac, 1, 0) AS supports_mac,
    IF(supports_linux, 1, 0) AS supports_linux,

    -- Multi-Hot Encoded Genres (En Popüler Türler)
    IF(REGEXP_CONTAINS(genres, r'action'), 1, 0) AS genre_action,
    IF(REGEXP_CONTAINS(genres, r'indie'), 1, 0) AS genre_indie,
    IF(REGEXP_CONTAINS(genres, r'casual'), 1, 0) AS genre_casual,
    IF(REGEXP_CONTAINS(genres, r'adventure'), 1, 0) AS genre_adventure,
    IF(REGEXP_CONTAINS(genres, r'simulation'), 1, 0) AS genre_simulation,
    IF(REGEXP_CONTAINS(genres, r'strategy'), 1, 0) AS genre_strategy,
    IF(REGEXP_CONTAINS(genres, r'rpg'), 1, 0) AS genre_rpg,

    -- Multi-Hot Encoded Categories (Kategoriler)
    IF(REGEXP_CONTAINS(categories, r'single-player'), 1, 0) AS cat_singleplayer,
    IF(REGEXP_CONTAINS(categories, r'multi-player|pvp'), 1, 0) AS cat_multiplayer,
    IF(REGEXP_CONTAINS(categories, r'co-op'), 1, 0) AS cat_coop,
    IF(REGEXP_CONTAINS(categories, r'family sharing'), 1, 0) AS cat_family_sharing,
    IF(REGEXP_CONTAINS(categories, r'full controller support|partial controller support'), 1, 0) AS cat_controller,

    -- Multi-Hot Encoded Tags (Etiketler)
    IF(REGEXP_CONTAINS(tags, r'sandbox'), 1, 0) AS tag_sandbox,
    IF(REGEXP_CONTAINS(tags, r'building'), 1, 0) AS tag_building,
    IF(REGEXP_CONTAINS(tags, r'atmospheric'), 1, 0) AS tag_atmospheric,
    IF(REGEXP_CONTAINS(tags, r'relaxing'), 1, 0) AS tag_relaxing,
    IF(REGEXP_CONTAINS(tags, r'horror'), 1, 0) AS tag_horror,
    IF(REGEXP_CONTAINS(tags, r'open world'), 1, 0) AS tag_open_world,

    -- Dil Desteği
    IF(REGEXP_CONTAINS(supported_languages, r'english'), 1, 0) AS lang_english,
    IF(REGEXP_CONTAINS(supported_languages, r'turkish'), 1, 0) AS lang_turkish

FROM stg