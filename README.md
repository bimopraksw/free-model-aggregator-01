# free-model-aggregator-01

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/bimopraksw/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/bimopraksw/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One zero-config server that serves the OpenAI Chat Completions API and the Anthropic Messages API over ten simulated free-tier providers. It probes every provider's catalog, ranks the free models by quality, routes each request to the best free model that meets its requirements (tools, vision, JSON mode, long context), speaks all three provider wires (OpenAI, Anthropic, Gemini) with lossless translation, and fails over to the next candidate when a provider refuses. `node main.ts --list` prints the catalog it found. Node 22 standard library only, nothing installed; the trial sets `MAKEFUN_ENV_URL` and `MAKEFUN_PROMPT_FILE`.

## Run it

```sh
node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=6vm00d4q&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
