import os

import httpx

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")


def get_bird_illustration(bird_species):
    # Use the OpenRouter API to generate a bird illustration based on the species
    prompt = f"Generate a detailed illustration of a {bird_species}."
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
    response = httpx.post(
        "https://api.openrouter.ai/v1/generate",
        json={"prompt": prompt},
        headers=headers,
    )

    if response.status_code == 200:
        return response.json().get("image_url")
    else:
        print(f"Error generating illustration: {response.text}")
        return None
