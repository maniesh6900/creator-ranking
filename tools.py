
import os
import json
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama  # swap for anthropic.Anthropic() if preferred

load_dotenv()

APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN")
if not APIFY_API_TOKEN:
    raise RuntimeError("APIFY_API_TOKEN is not set. Add it to your .env file.")
APIFY_ACTOR = "apify~instagram-scraper"
APIFY_BASE = "https://api.apify.com/v2"

llm_client = ChatOllama(model="gpt-oss:20b", format="json")


class VetCreatorInput(BaseModel):
    username: str = Field(description="Public Instagram username, without the @")
    product_description: str = Field(
        description="What you're promoting -- product, niche, target audience, brand tone. "
                    "e.g. 'Vegan protein powder for gym-goers aged 20-35, playful brand voice'"
    )


def _scrape_profile(username: str, timeout_s: int = 90) -> dict:
    """Run the Apify Instagram Scraper actor synchronously and return raw results."""
    run_input = {
        "directUrls": [f"https://www.instagram.com/{username}/"],
        "resultsType": "details",       # profile info + recent posts
        "resultsLimit": 3,              # last 3 posts/reels
        "searchType": "user",
    }

    # Run the actor and wait for it to finish (sync endpoint, up to `timeout_s`)
    run_url = f"{APIFY_BASE}/acts/{APIFY_ACTOR}/run-sync-get-dataset-items"
    resp = requests.post(
        run_url,
        params={"token": APIFY_API_TOKEN, "timeout": timeout_s},
        json=run_input,
        timeout=timeout_s + 15,
    )
    resp.raise_for_status()
    items = resp.json()

    if not items:
        raise ValueError(
            f"No data returned for '{username}'. Account may be private, "
            "deleted, or Instagram blocked this scrape attempt -- try again."
        )
    return items[0]  # profile object with nested latestPosts


def _classify_and_score(profile: dict, product_description: str) -> dict:
    """Single LLM call: categorize posts + produce a fit score with reasoning."""
    posts = profile.get("latestPosts", [])[:3]

    posts_summary = []
    total_likes, total_comments = 0, 0
    for p in posts:
        posts_summary.append({
            "caption": (p.get("caption") or "")[:300],
            "type": p.get("type"),  # Image / Video / Sidecar
            "likes": p.get("likesCount", 0),
            "comments": p.get("commentsCount", 0),
        })
        total_likes += p.get("likesCount", 0)
        total_comments += p.get("commentsCount", 0)

    followers = profile.get("followersCount", 0)
    engagement_rate = (
        round(((total_likes + total_comments) / max(len(posts), 1)) / max(followers, 1) * 100, 2)
        if followers else 0.0
    )

    prompt = f"""You are vetting an Instagram creator for a brand promotion.

CREATOR PROFILE:
Username: {profile.get('username')}
Bio: {profile.get('biography')}
Followers: {followers}
Avg engagement rate (based on last {len(posts)} posts): {engagement_rate}%

LAST {len(posts)} POSTS:
{json.dumps(posts_summary, indent=2)}

BRAND WANTS TO PROMOTE:
{product_description}

Return ONLY valid JSON with this exact shape:
{{
  "account_category": "broad category, e.g. Fitness, Beauty, Tech, Food, Parenting",
  "post_topics": ["short topic for post 1", "short topic for post 2", "short topic for post 3"],
  "fit_percent": <integer 1-100>,
  "reasoning": "2-3 sentences on why this score, covering category match, audience size, engagement, and content tone fit"
}}"""

    resp = llm_client.invoke([{"role": "user", "content": prompt}])
    result = json.loads(resp.content)
    result["followers_count"] = followers
    result["engagement_rate_percent"] = engagement_rate
    return result


@tool("vet_instagram_creator_for_promotion", args_schema=VetCreatorInput)
def vet_instagram_creator_for_promotion(username: str, product_description: str) -> dict:
    """
    Scrape a public Instagram creator's profile and assess whether they're
    a good fit to promote a given product. Returns account category,
    topics of their last 3 posts/reels, follower count, engagement rate,
    and a 1-100% fit score with reasoning.
    """
    profile = _scrape_profile(username)
    result = _classify_and_score(profile, product_description)
    result["username"] = username
    return result