# 🎧 Model Card: 🎵 Tunecraft

## Intended Use

Tunecraft is intended for individual listeners who want personalized, explained song recommendations based on structured taste (genre, mood, energy, acoustic preference) or a free-text description of a vibe. It is meant for casual, single-session discovery, not for production-scale music streaming or commercial licensing decisions. The tool is best suited for users exploring a bounded dataset of curated/public songs rather than the full commercial music catalog.

## Model Structure

Tunecraft is not a trained ML model; it's a rule-based content scorer combined with a retrieval-augmented pipeline. A `UserProfile` is scored against song metadata (genre, mood, energy, valence, danceability) using weighted feature matching, with an artist cap for diversity. Optional free-text input is embedded and matched via ChromaDB semantic search to narrow the candidate pool before scoring.

## Limitations and Biases

The system's accuracy is capped by the quality and coverage of its underlying CSV dataset, so niche genres or moods with few songs will yield weaker matches. Because scoring relies on hand-picked feature weights rather than learned preferences, it cannot adapt to individual listening history or nuanced taste over time. Semantic search quality also depends on the embedding model's ability to interpret informal, slang-heavy, or ambiguous vibe descriptions.

## Potential Misuses and Their Prevention

The tool could be misused to scrape or bulk-export the underlying song dataset rather than for its intended one-off recommendation use; it is not designed for high-volume automated querying. Adversarial or nonsensical inputs (invalid modes, out-of-range values) are sanitized and clamped so they cannot crash the app or produce misleading output. Since results are drawn from a public, non-authoritative dataset, they should not be treated as licensed music metadata or used for commercial cataloging.

---

## AI Collaboration

A lot of my interactions and collaboration with AI (Claude Code) during this project was based on system design. I often had a pre-existing idea in my mind that I presented to Claude and asked for its own thoughts and suggestions. This iterative feedback loop between me and the AI process helped me improve my overall design of the system and significantly helped me whenever I ran into any doubts. If I ever had a question, I would ask Claude to present me with design options with their pros and cons so that I could make the best choice for the system I was envisioning.

AI was very helpful with giving suggestions regarding the creation of the RAG pipeline, vector database, and semantic search features. I am not the most knowledgeable in these areas yet, but with the help of Claude I was able to make decisions such as what vector database I should use to support a large amount of songs.

AI did often struggle to remember the generated plan I was carrying throughout the same or multiple conversations. I created a 7 step execution plan after working on my design plan with Claude for about an hour. The generated text script had all the information needed to make the entire application in one go. However, I decided to move through the project one step at a time to ensure everything was working. When I got to the vector database portion of implementation, the AI kept suggesting to use an API key for an OpenAI model or the Spotify API despite the prior given decision of using RAG and a vector database to support the recommender system. I just had to correct it and continue on with my plan.

---

## Model Reliability Surprises

It was very difficult to get the model to fail obtaining recommended songs when testing the Streamlit app. The model always has enough scored songs to return a ranked list, even under adversarial profiles like invalid scoring modes, empty preferences, or contradictory inputs. This surprised me since I expected edge cases to require more explicit fallback handling; instead, the sanitization and scoring logic absorbed most of them gracefully. The one place reliability did dip was semantic search response time, which slowed noticeably as the RAG-loaded song pool grew larger.
