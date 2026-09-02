{% set genres_list = [
  'action', 'adventure', 'casual', 'rpg', 'simulation', 'strategy', 
  'indie', 'early access', 'free to play', 'massively multiplayer', 
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

WITH stg AS (
    SELECT * FROM {{ ref('stg_steam_games') }}
)

SELECT
    app_id,
    name,
    
    -- Target Variables
    price,                           -- Fiyat Tahmini (Regression)
    positive_review_percentage,      -- Review Skor Tahmini (%)
    estimated_owners_avg,            -- Oyuncu Sayısı Tahmini

    -- Basic Numeric Features
    release_year,
    achievements_count,
    dlc_count,
    COALESCE(screenshot_count, 0) AS screenshot_count,
    COALESCE(movie_count, 0) AS movie_count,
    has_website,
    has_support_url,
    has_support_email,
    
    -- Platforms
    IF(supports_windows, 1, 0) AS supports_windows,
    IF(supports_mac, 1, 0) AS supports_mac,
    IF(supports_linux, 1, 0) AS supports_linux,

    -- Dynamic Multi-Hot Encoding for Genres
    {% for genre in genres_list %}
    IF(REGEXP_CONTAINS(LOWER(genres), r'{{ genre }}'), 1, 0) AS genre_{{ genre | replace(' ', '_') | replace('&', 'and') }},
    {% endfor %}

    -- Dynamic Multi-Hot Encoding for Categories
    {% for category in categories_list %}
    IF(REGEXP_CONTAINS(LOWER(categories), r'{{ category }}'), 1, 0) AS cat_{{ category | replace(' ', '_') | replace('-', '_') | replace('/', '_') }},
    {% endfor %}

    -- Dynamic Multi-Hot Encoding for Key Tags
    {% for tag in tags_list %}
    IF(REGEXP_CONTAINS(LOWER(tags), r'{{ tag }}'), 1, 0) AS tag_{{ tag | replace(' ', '_') | replace('-', '_') }}{% if not loop.last %},{% endif %}
    {% endfor %}

FROM stg