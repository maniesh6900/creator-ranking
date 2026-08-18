# Instagram Creator Vetting Agent

A LangChain agent that vets Instagram creators for brand promotions. Given a list of
Instagram usernames, it scrapes each creator's public profile, classifies their content,
and scores how good a fit they are for a given product — returning a structured report
with a 1–100% fit score and reasoning.

## How it works

1. **Scraping** — Uses the [Apify Instagram Scraper](https://apify.com/apify/instagram-scraper)
   actor to pull each profile's details (bio, follower count) plus their last 3 posts/reels.
2. **Classification & scoring** — An LLM running locally via [Ollama](https://ollama.com)
   (`gpt-oss:20b`) categorizes the account, summarizes post topics, and produces a fit
   percentage with reasoning. Engagement rate is computed from likes/comments vs. followers.
3. **Agent loop** — `main.py` uses `langchain-classic`'s tool-calling agent, so the LLM
   decides when to call the vetting tool for each username and parses the results into a
   structured `Reacherreponse` model.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally with the `gpt-oss:20b` model pulled:
  ```bash
  ollama pull gpt-oss:20b
  ```
  The server must be reachable at `http://localhost:11434` (the script fails fast otherwise).
- An [Apify](https://apify.com) account with an API token (the Instagram Scraper actor
  charges usage credits).

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp example.env .env
# Then edit .env and set APIFY_API_TOKEN to your Apify token.

# 3. Make sure Ollama is running (desktop app or `ollama serve`)
ollama list   # should show gpt-oss:20b
```

## Usage

```bash
python main.py
```

You'll be prompted for Instagram usernames (comma-separated). For example:

```
Enter Instagram usernames to vet, comma-separated (e.g. maniesh6900, anotheruser): maniesh6900, somecreator
```

If you press Enter with no input, it defaults to `maniesh6900`. The hardcoded product in
`main.py` is the **Logitech G102 Light Sync Gaming Mouse** — edit the query in `main.py`
to vet creators against a different product.

Each username produces a report containing:

- `topic` — the product/topic evaluated
- `summary` — the vetting result (category, post topics, followers, engagement rate, fit %)
- `sources` — profile data used
- `tools_used` — which tools the agent invoked

## Project structure

```
main.py         Entry point: builds the agent, prompts for usernames, prints reports
tools.py        The vet_instagram_creator_for_promotion tool (scrape + classify + score)
requirements.txt
example.env     Template for environment variables (copy to .env)
```

## Notes

- Only **public** profiles can be scraped; private/deleted accounts raise an error.
- The scraper uses Apify credits per run.
- `APIFY_API_TOKEN` is required — the tool raises `RuntimeError` if it's missing.
