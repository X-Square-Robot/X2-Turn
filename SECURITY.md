# Security policy

Please report suspected vulnerabilities privately to the repository maintainers
or the security contact published by the hosting organization. Do not include
credentials, private audio, model tokens, personal data, or exploit details in
a public issue.

Only the latest revision of the default branch is supported. Reports should
include affected components, reproduction steps, impact, and suggested
mitigations when available.

## Deployment guidance

The browser demos are development tools. They do not provide production-grade
authentication, authorization, rate limiting, or durable upload isolation.

- Keep local services bound to `127.0.0.1` unless remote access is required.
- Use authenticated HTTPS termination, trusted certificates, network access
  controls, and request limits for public deployments.
- Treat uploaded audio, microphone audio, transcripts, and trace files as
  sensitive user data.
- Do not log, retain, or redistribute speech without appropriate consent.
- Run model, vLLM, LLM, and TTS services with least-privilege filesystem and
  network access.
- Never use the generated self-signed development certificate as production
  TLS.
