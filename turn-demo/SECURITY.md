# Security

The demo is a development tool and does not implement authentication,
authorization, rate limiting, or durable upload storage.

- Keep the default `127.0.0.1` bind address for local evaluation.
- Use authenticated HTTPS termination before exposing the service remotely.
- Treat uploaded and microphone audio as sensitive data.
- Do not log, retain, or redistribute speech without appropriate consent.
- Run model and vLLM services with least-privilege filesystem access.

Report vulnerabilities privately to the repository maintainers. Do not include
credentials, private audio, or personal data in an issue.
