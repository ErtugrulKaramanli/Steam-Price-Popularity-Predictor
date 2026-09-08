{% set genres_list = [
  'action', 'adventure', 'casual', 'rpg', 'simulation', 'strategy', 
  'indie', 'early access', 'massively multiplayer', 
  'racing', 'sports', 'design & illustration', 'animation & modeling', 
  'game development', 'education', 'utilities'
] %}

{% set categories_list = [
  'single-player', 'multi-player', 'co-op', 'pvp', 'mmo',
  'online pvp', 'online co-op', 'shared/split screen',
  'full controller support', 'partial controller support', 
  'vr only', 'vr supported',
  'steam achievements', 'steam cloud', 'steam trading cards', 
  'steam workshop', 'steam leaderboards',
  'in-app purchases', 'includes level editor', 'remote play together', 'family sharing'
] %}

{% set tags_list = [
  '2d', '3d', 'pixel graphics', 'anime', 'stylized', 'hand drawn', 'retro', 'cartoony', 'minimalist',
  'story rich', 'atmospheric', 'fantasy', 'sci fi', 'horror', 'psychological horror', 'dark', 'open world', 'space', 'post apocalyptic', 'cyberpunk',
  'roguelike', 'roguelite', 'metroidvania', 'souls like', 'deckbuilding', 'bullet hell', 'tower defense', 'city builder', 'survival', 'crafting', 'turn based', 'hack and slash', 'fps', 'walking simulator', 'visual novel', 'sandbox', 'management', 'puzzle platformer'
] %}

{% set languages_list = [
  'english', 'french', 'italian', 'german', 'spanish', 'arabic', 'bulgarian', 
  'portuguese', 'hungarian', 'greek', 'danish', 'traditional chinese', 
  'simplified chinese', 'korean', 'dutch', 'norwegian', 'polish', 'romanian', 
  'russian', 'thai', 'turkish', 'ukrainian', 'finnish', 'czech', 'swedish', 'japanese'
] %}

WITH stg AS (
    SELECT * FROM {{ ref('stg_steam_games') }}
)

SELECT
    app_id,
    name,
    
    -- Targets
    price,                           -- Fiyat Tahmini (Regression Target)
    positive_review_percentage,      -- Review Skor Tahmini (%)
    
    -- Homojen 5'li Sınıflandırma Target'ı (dbt Transformation)
    CASE 
        WHEN estimated_owners_raw IN ('0 - 20000', '0 .. 20,000') THEN '0-20k'
        WHEN estimated_owners_raw IN ('20000 - 50000', '20,000 .. 50,000') THEN '20k-50k'
        WHEN estimated_owners_raw IN (
            '50000 - 100000', '50,000 .. 100,000',
            '100000 - 200000', '100,000 .. 200,000'
        ) THEN '50k-200k'
        WHEN estimated_owners_raw IN (
            '200000 - 500000', '200,000 .. 500,000',
            '500000 - 1000000', '500,000 .. 1,000,000'
        ) THEN '200k-1M'
        ELSE '1M+'
    END AS target_owner_class,

    -- Basic Numeric Features
    release_year,
    achievements_count,
    dlc_count,
    screenshot_count,
    movie_count,
    (screenshot_count + movie_count) AS total_media_count,
    has_website,
    has_support_url,
    has_support_email,
    
    -- Platforms
    IF(supports_windows, 1, 0) AS supports_windows,
    IF(supports_mac, 1, 0) AS supports_mac,
    IF(supports_linux, 1, 0) AS supports_linux,

    -- Dynamic Multi-Hot Encoding for Genres
    {% for genre in genres_list %}
    IF(REGEXP_CONTAINS(genres, r'{{ genre }}'), 1, 0) AS genre_{{ genre | replace(' ', '_') | replace('&', 'and') }},
    {% endfor %}

    -- Dynamic Multi-Hot Encoding for Categories
    {% for category in categories_list %}
    IF(REGEXP_CONTAINS(categories, r'{{ category }}'), 1, 0) AS cat_{{ category | replace(' ', '_') | replace('-', '_') | replace('/', '_') }},
    {% endfor %}

    -- Dynamic Multi-Hot Encoding for Key Tags
    {% for tag in tags_list %}
    IF(REGEXP_CONTAINS(tags, r'{{ tag }}'), 1, 0) AS tag_{{ tag | replace(' ', '_') | replace('-', '_') }},
    {% endfor %}

    -- Dynamic Multi-Hot Encoding for Supported Languages
    {% for lang in languages_list %}
    IF(REGEXP_CONTAINS(supported_languages, r'{{ lang }}'), 1, 0) AS lang_{{ lang | replace(' ', '_') }}{% if not loop.last %},{% endif %}
    {% endfor %}

FROM stg