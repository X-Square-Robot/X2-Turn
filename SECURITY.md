# Security policy

Please report suspected vulnerabilities privately to [kaiqifu@x2robot.com](mailto:kaiqifu@x2robot.com).
Do not include credentials, private audio, model tokens, personal data, or
exploit details in a public issue.

Only the latest revision of the default branch is supported. Reports should
include affected components, reproduction steps, impact, and suggested
mitigations when available.

## Deployment guidance

The browser demos are development tools. They do not provide production-grade
authentication, authorization, rate limiting, or durable upload isolation.

Local defaults bind to `127.0.0.1`. Set `BIND_HOST=0.0.0.0`, `VOXTRAL_HOST`,
or `--host 0.0.0.0` only when another machine must connect. Docker Compose
still uses `0.0.0.0` inside the container network and publishes host ports.

- Use authenticated HTTPS termination, trusted certificates, network access
  controls, and request limits for public deployments.
- Treat uploaded audio, microphone audio, transcripts, and trace files as
  sensitive user data.
- Do not log, retain, or redistribute speech without appropriate consent.
- Run model, vLLM, LLM, and TTS services with least-privilege filesystem and
  network access.
- Never use the generated self-signed development certificate as production
  TLS.
